import numpy as np
import torch


def Iou(box,boxes,is_Min=True):
    '''

    '''
    box_area = (box[2]-box[0])*(box[3]-box[1])
    boxes_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    x1 = np.maximum(box[0],boxes[:,0])
    y1 = np.maximum(box[1],boxes[:,1])
    x2 = np.maximum(box[2],boxes[:,2])
    y2 = np.maximum(box[3],boxes[:,3])

    w = np.maximum(0,x2-x1)
    h = np.maximum(0,y2-y1)
    inter = w*h

    if is_Min:
        return  np.true_divide(inter,np.maximum(box_area,boxes_area))
    else:
        return np.true_divide(inter,(box_area+boxes_area-inter))

def giou(box,boxes):
    '''

    '''
    iou = Iou(box,boxes)
    box_area = (box[2] - box[0]) * (box[3] - box[1])

    '''
    计算最大面积
    '''
    max_x = np.maximum(max(box[2],box[0]),np.maximum(boxes[:,2],boxes[:,0]))
    min_x =  np.minimum(min(box[2],box[0]),np.minimum(boxes[:,2],boxes[:,0]))
    max_y = np.maximum(max(box[3], box[1]), np.maximum(boxes[:, 3], boxes[:, 1]))
    min_y = np.minimum(min(box[3], box[1]), np.minimum(boxes[:, 3], boxes[:, 1]))

    area_C = (max_x-min_x)*(max_y-min_y)


    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.maximum(box[2], boxes[:, 2])
    y2 = np.maximum(box[3], boxes[:, 3])

    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    Area  = w * h

    sum_area = np.add(box_area,boxes_area)

    add_area = sum_area - Area
    end_area = (area_C-add_area)/area_C
    giou = iou - end_area

    return giou


def diou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:  #
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]

    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)

    union = area1 + area2 - inter_area

    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)

    if exchange:
        dious = dious.T
    return dious


if __name__ == '__main__':

    box = np.array([1, 1, 3, 3])
    boxes = np.array([[1, 1, 3, 3], [3, 3, 5, 5], [2, 2, 4, 4]])
    y = giou(box, boxes)
    print(y)
