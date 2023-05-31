from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes, BoundingBox
import torch
from podm.box import Box, intersection_over_union
from podm.metrics import MetricPerClass
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES as classes



def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# define the training tranforms


def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })
# define the validation transforms


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def calculate_iou(box_a, box_b):

    A = box_a.size(0)
    B = box_b.size(0)

    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(
        A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(
        A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3] -
              box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3] -
              box_b[:, 1])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b - inter
    iou = torch.diag(inter / union)

    enclose_mins = torch.min(box_a[:, :2], box_b[:, :2])
    enclose_maxes = torch.max(box_a[:, 2:], box_b[:, 2:])
    enclose_wh = torch.max(enclose_maxes - enclose_mins,
                           torch.zeros_like(enclose_maxes))

    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    giou = iou - (enclose_area - torch.diag(union)) / enclose_area

    return iou, giou


def get_mIOU(pre_boxes, target_boxes, iou_threshold):
    mIOU_list_each = []
    matched_target = []
    for k in range(len(pre_boxes)):
        for j in range(len(target_boxes)):
            box1 = Box.of_box(
                pre_boxes[k][0], pre_boxes[k][1], pre_boxes[k][2], pre_boxes[k][3])
            box2 = Box.of_box(
                target_boxes[j][0], target_boxes[j][1], target_boxes[j][2],
                target_boxes[j][3])
            IOU = intersection_over_union(box1, box2)
            if IOU >= iou_threshold and j not in matched_target:
                matched_target.append(j)
                mIOU_list_each.append(IOU)
    mIOU = sum(mIOU_list_each) / (len(mIOU_list_each)+1e-5)
    return mIOU

def get_mAP(pre_class, pre_boxes, pre_scores, target_class, target_boxes, iou_threshold):

    pre_box = []
    for i in range(len(pre_class)):
        bb = BoundingBox.of_bbox(None, category=pre_class[i], xtl=pre_boxes[i][0], ytl=pre_boxes[i]
                                 [1], xbr=pre_boxes[i][2], ybr=pre_boxes[i][3], score=pre_scores[i])
        pre_box.append(bb)

    target_box = []
    for i in range(len(target_class)):
        bb = BoundingBox.of_bbox(None,
                                 category=target_class[i],
                                 xtl=target_boxes[i][0],
                                 ytl=target_boxes[i][1],
                                 xbr=target_boxes[i][2],
                                 ybr=target_boxes[i][3])
        target_box.append(bb)

    results = get_pascal_voc_metrics(target_box, pre_box, iou_threshold)
    tps_list = []
    for _, metric in results.items():
        tps_list.append(metric.tp)
    tp = sum(tps_list)/len(tps_list)
    acc = tp / (len(pre_class)+1e-5)
    mAP = MetricPerClass.mAP(results)
    return mAP, acc
