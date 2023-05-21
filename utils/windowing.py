import numpy as np
import os
from utils.general import xywhn2xyxy, scale_boxes, xyxy2xywh, xyxy2xywhn


def calc_shape(shape, W, S):
    ds = np.ceil(shape / S)
    '''
    print("ds", ds)
    print("if", ((np.ceil(shape / S) - 1) - 1) * S + W)
    print("elif", (np.ceil(shape / S) - 1) * S + W)
    print("ds//", np.ceil(shape / S) - 1)
    '''
    if ((np.ceil(shape / S) - 1) - 1) * S + W == shape:
        # ds = shape // S
        ds = np.ceil(shape / S) - 1
    elif (np.ceil(shape / S) - 1) * S + W >= shape:
        ds = np.ceil(shape / S)
    return ds


def change_view_func2D(arr, patch_shape=(2, 5), overlap=(0, 0)):
    D = arr.strides[1]  # datum size in bytes
    W1 = patch_shape[0]  # window size
    O1 = overlap[0]  # overlap
    S1 = W1 - O1  # step size
    W2 = patch_shape[1]  # window size
    O2 = overlap[1]  # overlap
    S2 = W2 - O2  # step size
    
    shape = (
        calc_shape(arr.shape[0], W1, S1).astype(int),
        calc_shape(arr.shape[1], W2, S2).astype(int),
        W1,
        W2,
    )

    strides = (S1 * arr.strides[0], S2 * arr.strides[1], arr.strides[0], arr.strides[1])

    res = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return res.reshape((shape[0] * shape[1], shape[2], shape[3]))


def change_view_func3D(arr, patch_shape=(2, 5), overlap=(0, 0)):
    D = arr.strides[-1]  # datum size in bytes
    W1 = patch_shape[0]  # window size
    O1 = overlap[0]  # overlap
    S1 = W1 - O1  # step size
    W2 = patch_shape[1]  # window size
    O2 = overlap[1]  # overlap
    S2 = W2 - O2  # step size
    '''
    print("D", D)
    print("W1", W1)
    print("O1", O1)
    print("S1", S1)
    print("W2", W2)
    print("O2", O2)
    print("S2", S2)
    '''
    shape = (
        calc_shape(arr.shape[1], W1, S1).astype(int),
        calc_shape(arr.shape[2], W2, S2).astype(int),
        arr.shape[0],
        W1,
        W2,
    )
    # print("shape", shape)
    strides = (
        S1 * arr.strides[1],
        S2 * arr.strides[2],
        arr.strides[0],
        arr.strides[1],
        arr.strides[2],
    )
    # print("strides", strides)
    res = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    # print("res_strided", res.shape)
    return res.reshape((shape[0] * shape[1], shape[2], shape[3], shape[4])), shape[:2]


def large_image_to_batch(img, patch_shape, overlap):
    step_size = (patch_shape[0] - overlap[0], patch_shape[1] - overlap[1])
    if (((np.ceil(img.shape[1] / step_size[0]) - 1) * step_size[0]) + overlap[0]) == img.shape[1]:
        padded_shape_height = img.shape[-2]
    else:
        padded_shape_height = ((np.ceil(img.shape[1] / step_size[0]).astype(int) - 1) * step_size[0]) + patch_shape[0]
        
    if (((np.ceil(img.shape[2] / step_size[1]) - 1) * step_size[1]) + overlap[1]) == img.shape[2]:
        padded_shape_width = img.shape[-1]
    else:
        padded_shape_width = ((np.ceil(img.shape[2] / step_size[1]).astype(int) - 1) * step_size[1]) + patch_shape[1]
    
    padded_shape = (padded_shape_height, padded_shape_width)

    # print(f"patch_shape {patch_shape}")
    # print(f"step size {step_size}")
    # print(f"overlap {overlap}")
    
    # print("padded image shape before", padded_shape)

    assert padded_shape >= img.shape

    if padded_shape > img.shape:
        pad_x = (0, padded_shape[1] - img.shape[-1])
        pad_y = (0, padded_shape[0] - img.shape[-2])
        # print("pads", pad_x, pad_y)
        img = np.pad(img, ((0, 0), pad_y, pad_x))
    else:
        pad_x = pad_y = None

    # print("padded image shape ", img.shape)

    batched, grid_shape = change_view_func3D(img, patch_shape, overlap)

    # print(f"grid_shape {grid_shape}")
    # print(f"batched {batched.shape}")

    return batched, (pad_y, pad_x), grid_shape


def large_image_to_batch_simple(img, patch_shape, overlap):
    step_size = (patch_shape[0] - overlap[0], patch_shape[1] - overlap[1])
    padded_shape = (
        ((np.ceil(img.shape[1] / step_size[0]).astype(int) - 1) * step_size[0])
        + patch_shape[0],
        ((np.ceil(img.shape[2] / step_size[1]).astype(int) - 1) * step_size[1])
        + patch_shape[1],
    )
    # print(f"patch_shape {patch_shape}")
    # print(f"step size {step_size}")
    # print(f"overlap {overlap}")

    assert padded_shape >= img.shape

    # print(img.shape)
    # print(padded_shape)

    if padded_shape > img.shape:
        pad_x = (0, padded_shape[1] - img.shape[1])
        pad_y = (0, padded_shape[0] - img.shape[0])
        img = np.pad(img, ((0, 0), pad_y, pad_x))
    else:
        pad_x = pad_y = None

    xd = ((padded_shape[1] - patch_shape[1]) / step_size[1]) + 1
    yd = ((padded_shape[0] - patch_shape[0]) / step_size[0]) + 1
    res_shape = (
        xd * yd,
        3,
        patch_shape[0],
        patch_shape[1],
    )

    res_img = np.zeros(res_shape)

    for i in range(yd):
        for j in range(xd):
            res_img[i * xd + j, :, :, :] = img[
                :,
                i * patch_shape[0] : (i + 1) * patch_shape[0],
                j * patch_shape[1] : (j + 1) * patch_shape[1],
            ]

    return res_img, (pad_y, pad_x)


def rescale_coords_and_save(elems, i, label_path, grid_shape=(3, 4), old_shape=(1200, 1920), step_size=(480, 480), new_shape=(640, 640), patches_labels_path="/workspace8/yolo_train_data/1401_New_Organization_for_Datasets/train_summer_1401/Patches/yolo_labels_with_unk_as_bird/"):
            
    new_elems = [float(elem) for elem in elems[1: 5]]
    isunknown = elems[-1]
    
    new_elems = np.asarray([new_elems])
    
    new_elems[:, : 4] = xywhn2xyxy(new_elems[:, : 4], w=old_shape[1], h=old_shape[0])
    
    
    offset_coeffs = (
        np.mgrid[0 : grid_shape[0] : 1, 0 : grid_shape[1] : 1]
        .reshape(2, -1)
        .T
    )
    offsets = offset_coeffs * np.array(step_size)  # y, x offsets
    
    new_elems[:, [1, 0]] -= offsets[i]  # y0, x0 padding
    new_elems[:, [3, 2]] -= offsets[i]  # y1, x1 padding
    
    #new_elems[:, : 4] = scale_boxes(
    #    old_shape, new_elems[:, : 4], new_shape
    #)
    
    new_elems[:, : 4] = xyxy2xywhn(new_elems[:, : 4])
    
    new_elems = [str(elem) for elem in new_elems[0]]
    
    new_elems.insert(0, str(0))
    new_elems.append(isunknown)
    
    line = " ".join(new_elems)
    
    with open(os.path.splitext(patches_labels_path + label_path.split('/')[-1])[0] + "_" + str(i) + ".txt" , "a") as fp:
        fp.write(line)
        fp.write("\n")