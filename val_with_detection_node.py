from PIL import Image
from time import time
import glob

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from detection_node import Inference_bird, detect, detect_with_FI
from models.experimental import attempt_load
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study, plot_images_tp_fp
from utils.torch_utils import select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    plot_matches = []
    
    for i in range(len(iouv)):
        x = torch.where((iou > iouv[i]) & correct_class)  # IoU > threshold and classes match # >= --> >
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            
            correct[matches[:, 1].astype(int), i] = True
            if i == 0:
                plot_matches = matches[matches[:, 2].argsort()[::-1]]
            # correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv # in old version
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device), plot_matches

@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.5,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        save_dir=Path(''),
        plots=True,
        test_only_coco_pretrained=False,
        preds_exist=False,
        preds_path='',
        FI = False, # detect with full inference
        callbacks=Callbacks(),
):
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    device = select_device(device, batch_size=batch_size)
    model = attempt_load(weights, device)  # load FP32 model
    
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    
    infer_obj = Inference_bird(
        model=model,
        imgsz=640,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )
    
    data = check_dataset(data)  # check
    
    # Configure
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 12, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    plot_iouv = torch.linspace(0.0, 0.9, 10, device=device)  # iou vector for plot 0.0:0.9
    plot_niou = plot_iouv.numel()
    
    # Dataloader
    task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataloader = create_dataloader(data[task],
                                   imgsz,
                                   batch_size,
                                   1,
                                   single_cls,
                                   pad=0,
                                   rect=True,
                                   prefix=colorstr(f'{task}: '))[0]
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile()  # profiling times
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    
    for batch_i, (img, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        
        plot_stats = []
        
        img = np.array(img)
        
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        
        out = []
        
        if preds_exist:
            for f in paths:
                in_file = open(preds_path + f.split("/")[-1].split(".")[0] + ".txt", 'r')
                preds = []
                for line in in_file.readlines():
                    pred = line.split(" ")
                    pred[-1] = 0
                    pred = [float(p) for p in pred]
                    pred[-1] = 0
                    preds.append(pred)
                out.append(torch.from_numpy(np.array(preds)))
                
        else:
            total_time = 0
            for im in img:
                im = im.transpose((1, 2, 0))
                with dt:
                    ftic = time()
                    if FI:
                        pred = detect_with_FI(im, infer_obj)
                    else:
                        pred = detect(im, infer_obj)
                    ftoc = time()
                print(f"One Image Full Time(s): {ftoc-ftic}")
                
                if test_only_coco_pretrained:
                    out.append(torch.from_numpy(np.array([p for p in pred[0] if p[5] == 14])))
                else:
                    pred = torch.from_numpy(pred)
                    out.append(pred[0])
                
                total_time += ftoc-ftic
            print(f"Detection Average Time(s): {total_time / len(img)}")
        
        '''
        out = torch.cat(out, dim=1)
        print(out.shape)
        
        print(out)
        out = out.cpu().numpy()
        print(out.shape)
        out = np.vstack(out)
        print(out.shape)
        out = torch.from_numpy(out)
        '''
        
        if test_only_coco_pretrained:
            targets[:, 1] += 14 # for birds
        
        ############################
        # Statistics per image
        for si, pred in enumerate(out):
            pred = pred.to(device)
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            plot_correct = torch.zeros(npr, plot_niou, dtype=torch.bool, device=device) # for plot tp fp
            seen += 1
            
            #print(img[si].shape)
            #print(shapes[si])
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    plot_stats.append((plot_correct, *torch.zeros((2, 0), device=device), labels[:, 0])) # for plot tp fp
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                '''
                print("predn")
                print(predn.shape)
                print("labelsn")
                print(labelsn.shape)
                '''
                correct, _ = process_batch(predn, labelsn, iouv)
                plot_correct, plot_matches = process_batch(predn, labelsn, plot_iouv) # for plot tp fp
                
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
                
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
            plot_stats.append((plot_correct, plot_matches))  # (plot_correct, plot_matches)
            
            
            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(img, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images_tp_fp(plot_stats, img, output_to_target(out), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if nc < 50 and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = dt.t / seen * 1E3  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(f'Speed: %.1fms inference per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = "/workspace8/yolo_train_data/1401_New_Organization_for_Datasets/test/custom_test_pycoco.json"  # annotations json (id = file_name and ann image_id = file_name)
        pred_json = str(save_dir / f'{w}_predictions.json')  # predictions
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools>=2.0.6')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            '''
            print(anno.dataset['images'])
            for img in anno.dataset['images']:
                img['id'] = img['file_name'].split('.')[0]
            for ann in anno.dataset['annotations']:
                ann['image_id'] = anno.dataset['images'][ann['image_id']-1]['id']
            print(anno.dataset['images'])
            with open("/workspace8/yolo_train_data/1401_New_Organization_for_Datasets/test/custom_test_pycoco.json", 'w') as f:
                json.dump(anno.dataset, f)
            print(anno.getImgIds())
            print("imgs_keys")
            print(anno.imgs.keys())
            '''
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            # map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, _), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--test-only-coco-pretrained', action='store_true', help='treat as only-coco-pretrained')
    parser.add_argument('--preds-exist', action='store_true', help='existing predictions ok, do not predict')
    parser.add_argument('--FI', action='store_true', help='detect with full inference')
    parser.add_argument('--preds-path', type=str, default=ROOT / '/workspace8/yolo_train_data/COCO_and_1401NOfD/coco_format/DETR_test2017_preds/', help='predictions path (default DETR)')
    #parser.add_argument('--augment', action='store_true', help='augmented inference')
    #parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    
    opt.data = check_yaml(opt.data)  # check YAML
    #opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            subprocess.run(['zip', '-r', 'study.zip', 'study_*.txt'])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
