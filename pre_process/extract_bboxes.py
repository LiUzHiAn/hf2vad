import numpy as np
import os
import argparse
import cv2
import torch
from tqdm import tqdm
from datasets.dataset import get_dataset, img_tensor2numpy, img_batch_tensor2numpy
from pre_process.mmdet_utils import init_detector, inference_detector

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

DATASET_CFGS = {
    "ped2": {"conf_thr": 0.5, "min_area": 10 * 10, "cover_thr": 0.6, "binary_thr": 18, "gauss_mask_size": 3, 'contour_min_area': 10 * 10},
    "avenue": {"conf_thr": 0.25, "min_area": 40 * 40, "cover_thr": 0.6, "binary_thr": 18, "gauss_mask_size": 5, 'contour_min_area': 40 * 40},
    "shanghaitech": {"conf_thr": 0.5, "min_area": 8 * 8, "cover_thr": 0.65, "binary_thr": 15, "gauss_mask_size": 5, 'contour_min_area': 40 * 40}
}


def getObjBboxes(img, model, dataset_name):
    result = inference_detector(model, img)
    CONF_THR = DATASET_CFGS[dataset_name]["conf_thr"]
    MIN_AREA = DATASET_CFGS[dataset_name]["min_area"]

    # bboxes = show_result(img, result, model.CLASSES, score_thr)
    bbox_result = result
    bboxes = np.vstack(bbox_result)

    scores = bboxes[:, -1]  # x1,y1,x2,y2,class_score
    bboxes = bboxes[scores > CONF_THR, :]

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    bbox_areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    return bboxes[bbox_areas >= MIN_AREA, :4]


def delCoverBboxes(bboxes, dataset_name):
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 4

    COVER_THR = DATASET_CFGS[dataset_name]["cover_thr"]

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    bbox_areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    sort_idx = bbox_areas.argsort()  # Index of bboxes sorted in ascending order by area size

    keep_idx = []
    for i in range(sort_idx.size):  # calculate overlap with i-th bbox
        # Calculate the point coordinates of the intersection
        x11 = np.maximum(x1[sort_idx[i]], x1[sort_idx[i + 1:]])
        y11 = np.maximum(y1[sort_idx[i]], y1[sort_idx[i + 1:]])
        x22 = np.minimum(x2[sort_idx[i]], x2[sort_idx[i + 1:]])
        y22 = np.minimum(y2[sort_idx[i]], y2[sort_idx[i + 1:]])
        # Calculate the intersection area
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h

        ratios = overlaps / bbox_areas[sort_idx[i]]
        num = ratios[ratios > COVER_THR]
        if num.size == 0:
            keep_idx.append(sort_idx[i])

    return bboxes[keep_idx]


def getFgBboxes(cur_img, img_batch, bboxes, dataset_name):
    area_thr = DATASET_CFGS[dataset_name]["contour_min_area"]
    binary_thr = DATASET_CFGS[dataset_name]["binary_thr"]
    gauss_mask_size = DATASET_CFGS[dataset_name]["gauss_mask_size"]
    extend = 2

    sum_grad = 0
    for i in range(img_batch.shape[0] - 1):
        img1 = img_batch[i, :, :, :]
        img2 = img_batch[i + 1, :, :, :]
        img1 = cv2.GaussianBlur(img1, (gauss_mask_size, gauss_mask_size), 0)
        img2 = cv2.GaussianBlur(img2, (gauss_mask_size, gauss_mask_size), 0)

        grad = cv2.absdiff(img1, img2)
        sum_grad = grad + sum_grad

    sum_grad = cv2.threshold(sum_grad, binary_thr, 255, cv2.THRESH_BINARY)[1]  # temporal gradient

    for bbox in bboxes:
        bbox_int = bbox.astype(np.int32)
        extend_y1 = np.maximum(0, bbox_int[1] - extend)
        extend_y2 = np.minimum(bbox_int[3] + extend, sum_grad.shape[0])
        extend_x1 = np.maximum(0, bbox_int[0] - extend)
        extend_x2 = np.minimum(bbox_int[2] + extend, sum_grad.shape[1])
        sum_grad[extend_y1:extend_y2 + 1, extend_x1:extend_x2 + 1] = 0

    sum_grad = cv2.cvtColor(sum_grad, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(sum_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg_bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        sum_grad = cv2.rectangle(sum_grad, (x, y), (x + w, y + h), color=255, thickness=1)
        area = (w + 1) * (h + 1)
        if area > area_thr and w / h < 10 and h / w < 10:
            extend_x1 = np.maximum(0, x - extend)
            extend_y1 = np.maximum(0, y - extend)
            extend_x2 = np.minimum(x + w + extend, sum_grad.shape[1])
            extend_y2 = np.minimum(y + h + extend, sum_grad.shape[0])
            fg_bboxes.append([extend_x1, extend_y1, extend_x2, extend_y2])

    return np.array(fg_bboxes)


def obj_bboxes_extraction(dataset_root, dataset_name, mode):
    # mmdet config file and pre-trained model weights
    mm_det_config_file = 'assets/latest_version_cascade_rcnn_r101_fpn_1x.py'
    mm_det_ckpt_file = 'assets/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth'

    dataset = get_dataset(dataset_name=dataset_name,
                          dir=os.path.join(dataset_root, dataset_name),
                          context_frame_num=1, mode=mode)

    mm_det_model = init_detector(mm_det_config_file, mm_det_ckpt_file, device="cuda:0")

    all_bboxes = list()

    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        batch, _ = dataset.__getitem__(idx)

        # centric frame
        cur_img = img_tensor2numpy(batch[1])
        h, w = cur_img.shape[0], cur_img.shape[1]

        obj_bboxes = getObjBboxes(cur_img, mm_det_model, dataset_name)
        # filter some overlapped bbox
        obj_bboxes = delCoverBboxes(obj_bboxes, dataset_name)
        fg_bboxes = getFgBboxes(cur_img, img_batch_tensor2numpy(batch), obj_bboxes, dataset_name)
        if fg_bboxes.shape[0] > 0:
            cur_bboxes = np.concatenate((obj_bboxes, fg_bboxes), axis=0)
        else:
            cur_bboxes = obj_bboxes

        all_bboxes.append(cur_bboxes)

    np.save(os.path.join(os.path.join(dataset_root, dataset_name),
                         '%s_bboxes_%s.npy' % (dataset_name, mode)), all_bboxes)
    print('bboxes saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_root", type=str, default="/home/liuzhian/hdd4T/code/hf2vad", help='project root path')
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--mode", type=str, default="train", help='train or test data')
    args = parser.parse_args()

    obj_bboxes_extraction(dataset_root=os.path.join(args.proj_root, "data"),
                          dataset_name=args.dataset_name,
                          mode=args.mode)