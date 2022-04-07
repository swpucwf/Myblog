import numpy as np


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






if __name__ == '__main__':

    box = np.array([1, 1, 3, 3])
    boxes = np.array([[1, 1, 3, 3], [3, 3, 5, 5], [2, 2, 4, 4]])
    y = giou(box, boxes)
    print(y)
