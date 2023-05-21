"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import sys
from pathlib import Path
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import cv2
from time import time

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
    set_logging,
    xywh2xyxy,
    xyxy2xywh
)
from utils.plots import colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox, letterbox_batch
from utils.windowing import large_image_to_batch

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert (
        im.data.contiguous
    ), "Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image."
    tl = (
        line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
    )  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            im,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

class Inference_bird:
    def __init__(
        self,
        model=None,
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        overlap_thres=0.25,
        half=False,
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.overlap_thres = overlap_thres

        # Initialize
        set_logging()
        self.device = select_device(0)
        self.half = half & (
            self.device.type != "cpu"
        )  # half precision only supported on CUDA

        # Load model
        if model == None:
            #self.model = attempt_load("yolov5x6.pt", map_location=self.device)  # load FP32 model
            self.model = attempt_load("yolov5n.pt", map_location=self.device)  # load FP32 model
        else:
            self.model = model
        # self.stride = int(self.model.stride.max())  # model stride
        # self.img_size = check_img_size(imgsz, s=self.stride)  # check image size
        self.img_size = imgsz
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )  # get class names
        if self.half:
            self.model.half()  # to FP16

        # Run inference
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.img_size, self.img_size)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )  # run once

    def img_to_batch(self, orig_img):
        img = orig_img.copy()
        #print("batching")
        patch_shape = (640, 640)  # (self.img_size, self.img_size)
        if img.shape[:2] <= patch_shape:
            return img, (0, 0), (0, 0), (1, 1)

        assert (
            img.shape[:2] > patch_shape
        ), f"input image must be larger or equal {patch_shape[0]}x{patch_shape[0]} got {img.shape[0]}x{img.shape[1]}"

        # replace 640 with self.img_size when retrained with correct training image sizes
        overlap = (
            np.ceil(640 * self.overlap_thres).astype(int), # 0.125 for 1200 height
            np.ceil(640 * self.overlap_thres).astype(int), # 0.25 for 1600 width and 0 for 1920
        )
        step_size = (patch_shape[0] - overlap[0], patch_shape[1] - overlap[1])

        img = img.transpose((2, 0, 1))

        images, pad_shape, grid_shape = large_image_to_batch(
            img,
            patch_shape,
            overlap,
        )  # windowing with p% overlap
        
        img = images.transpose((0, 2, 3, 1))

        return img, pad_shape, step_size, grid_shape
    
    def resize(self, orig_img, pp_img_size=640):
        pp_img = orig_img.copy()
        if len(pp_img.shape) == 3:
            h0, w0 = pp_img.shape[:2]  # orig hw
            r = pp_img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                pp_img = cv2.resize(
                    pp_img,
                    (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_AREA
                    if r < 1 #and not self.augment
                    else cv2.INTER_LINEAR,
                )
            # Padded resize
            pp_img = letterbox(pp_img, pp_img_size, stride=32)[0]
        else:
            res_img = letterbox_batch(pp_img, pp_img_size, stride=32)[0]
            for i in range(pp_img.shape[0]):
                assert (res_img[i] == letterbox(pp_img[i], pp_img_size, stride=32)[0]).all()
            pp_img = res_img
        
        return pp_img
    
    def pre_process(self, orig_img, pp_img_size=640, FI=False):
        print("pre-process")
        pp_img = orig_img.copy()
        if FI:
            pp_img = self.resize(pp_img, pp_img_size)
        
        # Convert
        if len(pp_img.shape) == 3:
            pp_img = pp_img.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB

            pp_img = np.expand_dims(pp_img, 0)
        else:
            temp = pp_img.copy()
            pp_img = pp_img.transpose((0, 3, 1, 2))
            '''
            for i in range(pp_img.shape[0]):
                pp_img[i] = temp[i].transpose((2, 0, 1))
                assert (pp_img[i] == temp[i].transpose((2, 0, 1))).all()
            '''

        pp_img = np.ascontiguousarray(pp_img)

        pp_img = torch.from_numpy(pp_img).to(self.device)
        pp_img = pp_img.half() if self.half else pp_img.float()  # uint8 to fp16/32
        pp_img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        return pp_img

    def predict(self, pp_img):
        with torch.no_grad():
            # Inference
            pred = self.model(pp_img)
            
            # print("***********predict*******", pred, pred.size(), sep="\n")
            # Apply NMS
            #print("prediction size: \n", pred.size())
            '''
            pred1 = nms_prep(pred, self.conf_thres, multi_label=False)
            print(pred1)
            #print(pred)
            #print(len(pred))
            #print(pred[0].shape)
            #pred = torch.cat(pred) My thinking is wrong
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                classes=None,
                agnostic=False,
                multi_label=True,
                max_det=self.max_det,
                merge=True
            )
            # print("***********nmspred*******", pred, pred.shape, sep="\n")
            commented by me
            '''
            return pred

    def post_process(self, pre_processed_ims, pred, orig_ims, step_size=None, grid_shape=None):
        # test detection for evaluation of reverse windowing process
        
        # Process detections
        pred = pred.cpu().numpy() 
        ''' top line added by me '''
        
        for i, det in enumerate(pred):  # detections per image
            
            if isinstance(orig_ims, list) or len(orig_ims.shape) > 3:  # batch_size >= 1
                orig_im = orig_ims[i].copy()
            else:
                orig_im = orig_ims.copy()

            # print(det.shape)
            
            if len(det):
                # Rescale boxes from img_size to orig_im size
                ''' changed by me '''
                pred[i][:, :4] = scale_boxes(
                    pre_processed_ims.shape[2:], xywh2xyxy(det[:, :4]), orig_im.shape
                )

                if torch.is_tensor(pred[i]):
                    pred[i] = pred[i].cpu().numpy()
                    
                if step_size is not None and grid_shape is not None:
                    offset_coeffs = (
                        np.mgrid[0 : grid_shape[0] : 1, 0 : grid_shape[1] : 1]
                        .reshape(2, -1)
                        .T
                    )
                    offsets = offset_coeffs * np.array(step_size)  # y, x offsets
                    pred[i][:, [1, 0]] += offsets[i]  # y0, x0 padding
                    pred[i][:, [3, 2]] += offsets[i]  # y1, x1 padding
                
                pred[i][:, :4] = xyxy2xywh(pred[i][:, :4])
                '''
                for j in range(len(pred[i])):
                    if pred[i][j][5] != 14:
                        pred[i][j] = np.zeros(shape=(0, 6), dtype=np.float32)
                '''
            else:
                pred[i] = pred[i].cpu().numpy()
            
        # if step_size is not None and grid_shape is not None:
        #     pred = np.vstack(pred)
        #     offset_coseffs = (
        #         np.mgrid[0 : grid_shape[0] : 1, 0 : grid_shape[1] : 1].reshape(2, -1).T
        #     )
        #     offsets = offset_coseffs * np.array(step_size)  # x,y offsets
        #     pred[:, [0, 1]] += offsets  # x padding
        #     pred[:, [2, 3]] += offsets  # y padding

        return pred
    
    def batch_pred_to_sngl(self, pred): # pred = tensor with 3 dimensions (1, predicts_number (-1), 6)
        #tic = time()    
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            classes=None,
            agnostic=False,
            multi_label=False,
            max_det=self.max_det,
            merge=True
        )
        #toc = time()
        #print(f"One Image NMS Time(s): {toc-tic}")
        
        pred = [p.cpu().numpy() for p in pred]
        print("conf mean :", np.mean(np.array(list(map(lambda p: p[4], pred[0])))))
        return pred
    
    def vis(self, im0, pred):
        im0 = np.ascontiguousarray(im0)
        # Write results
        for _, det in enumerate(pred):
            for *xyxy, conf, cls in reversed(det):  # Add bbox to image
                c = int(cls)  # integer class
                #label = self.names[c]
                label = f'{conf:.2f}'

                plot_one_box(
                    xyxy,
                    im0,
                    label=label,
                    color=colors(c, True),
                    line_thickness=2,
                )

        return im0


def detect(im, infer_obj):
    img_batch, pad_shape, step_size, grid_shape = infer_obj.img_to_batch(im)
            
    img_batch_t = infer_obj.pre_process(img_batch)
    
    pred = infer_obj.predict(img_batch_t)[0]
            
    pred = infer_obj.post_process(img_batch_t, pred, img_batch, step_size, grid_shape)
    
    pred = np.vstack(pred)
    pred = torch.from_numpy(pred)
    
    pred = infer_obj.batch_pred_to_sngl(pred[None, :])
            
    return np.asarray(pred)


def detect_with_FI(im, infer_obj):
    img_batch, pad_shape, step_size, grid_shape = infer_obj.img_to_batch(im)
            
    img_batch_t = infer_obj.pre_process(img_batch)
    orig_img_t = infer_obj.pre_process(im, max(im.shape[:2]), FI=True)
            
    pred_batch_t = infer_obj.predict(img_batch_t)[0]
    pred_orig_img_t = infer_obj.predict(orig_img_t)[0]
            
    pred_batch_t = infer_obj.post_process(img_batch_t, pred_batch_t, img_batch, step_size, grid_shape)
    pred_orig_img_t = infer_obj.post_process(orig_img_t, pred_orig_img_t, im)
    
    pred = torch.unsqueeze(torch.cat((torch.from_numpy(pred_batch_t).reshape(-1, 6), torch.from_numpy(pred_orig_img_t).reshape(-1,6))), 0)
    
    pred = infer_obj.batch_pred_to_sngl(pred)
            
    return np.asarray(pred)
    

def detect_with_FI_MS(im, infer_obj):
    
    img_batch, pad_shape, step_size, grid_shape = infer_obj.img_to_batch(im)
    img_batch_t = infer_obj.pre_process(img_batch)
    pred_batch_t = infer_obj.predict(img_batch_t)[0]
    pred_batch_t = infer_obj.post_process(img_batch_t, pred_batch_t, img_batch, step_size, grid_shape)
    
    im_mid_scale = infer_obj.resize(im, int(max(im.shape) / 2))
    img_batch_mid_scale, pad_shape_mid_scale, step_size_mid_scale, grid_shape_mid_scale = infer_obj.img_to_batch(im_mid_scale)
    img_batch_mid_scale_t = infer_obj.pre_process(img_batch_mid_scale)
    pred_batch_mid_scale_t = infer_obj.predict(img_batch_mid_scale_t)[0]
    pred_batch_mid_scale_t = infer_obj.post_process(img_batch_mid_scale_t, pred_batch_mid_scale_t, img_batch_mid_scale, step_size_mid_scale, grid_shape_mid_scale)
    
    orig_img_t = infer_obj.pre_process(im, max(im.shape[:2]), FI=True)
    pred_orig_img_t = infer_obj.predict(orig_img_t)[0]
    pred_orig_img_t = infer_obj.post_process(orig_img_t, pred_orig_img_t, im)
    
    pred = torch.unsqueeze(torch.cat((torch.from_numpy(pred_batch_t).reshape(-1, 6), torch.from_numpy(pred_batch_mid_scale_t).reshape(-1, 6), torch.from_numpy(pred_orig_img_t).reshape(-1,6))), 0)
    pred = infer_obj.batch_pred_to_sngl(pred)
            
    return np.asarray(pred)


def detect_v6(im, infer_obj):
    img_batch, pad_shape, step_size, grid_shape = infer_obj.img_to_batch(im)
            
    img_batch_t = infer_obj.pre_process(img_batch)
            
    pred = infer_obj.predict(img_batch_t)
            
    pred = infer_obj.post_process(img_batch_t, pred, img_batch, step_size, grid_shape)
    
    pred = np.vstack(pred)
    pred = torch.from_numpy(pred)
    
    pred = infer_obj.batch_pred_to_sngl(pred[None, :])
            
    return np.asarray(pred)        

'''
def detect(im, infer_obj):
            tic = time()
            img_batch, pad_shape, step_size, grid_shape = infer_obj.img_to_batch(im)
            #print(img_batch.shape, pad_shape, step_size, grid_shape)
            toc = time()
            print(f"One Image img_to_batch Time(s): {toc-tic}")
            
            tic = time()
            img_batch_t = infer_obj.pre_process(img_batch)
            toc = time()
            print(f"One Image pre_process Time(s): {toc-tic}")
            
            tic = time()
            pred = infer_obj.predict(img_batch_t)
            toc = time()
            print(f"One Image Predict Time(s): {toc-tic}")
            
            tic = time()
            pred = infer_obj.post_process(pred, img_batch, step_size, grid_shape)
            toc = time()
            print(f"One Image post_process Time(s): {toc-tic}")
            
            tic = time()
            pred = infer_obj.batch_pred_to_sngl(pred)
            toc = time()
            print(f"One Image batch_pred_to_sngl Time(s): {toc-tic}")
            
            return np.asarray(pred)
'''