import time

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image


def compare_embedding(embedding, facebank, threshold=1.2):
    if len(facebank) == 0:
        return None
    embedding = np.array(embedding)
    facebank = np.array(facebank).squeeze(axis=1)
    diff = embedding - facebank
    dist = np.sum(np.square(diff), axis=1)
    min_idx = np.argmin(dist)
    print(dist)
    if dist[min_idx] < threshold:
        return min_idx
    return None


def show_bboxes(img, bounding_boxes, facial_landmarks, names):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # print(1)
    """ Draw bounding boxes and facial landmarks. """
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "fonts/simhei.ttf", 20, encoding="utf-8")
    # 绘制文本
    # draw.text(position, text, textColor, font=fontStyle)
    bounding_boxes = np.asarray(bounding_boxes, dtype=int)
    for b, n in zip(bounding_boxes, names):
        # print(1)
        # cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
        # draw.rectangle((b[0], b[1]), (b[2], b[3]),(0, 0, 255), 1)
        draw.rectangle(((b[0], b[1]), (b[2], b[3])), outline="red", width=2)
        draw.text((int(b[0]), int(b[1])), n, (0, 0, 255), font=fontStyle)
        # cv2.putText(img, n, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    # 转换
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    for p in facial_landmarks:
        for i in range(5):
            cv2.circle(img, (int(p[i]), int(p[i + 5])), 2, color=(0, 255, 0), thickness=-1)
    return img


# def new_excel():
#     workbook = openpyxl.Workbook()
#     sheet = workbook.create_sheet(title='打卡记录')
#     del workbook['Sheet']
#
#     sheet["A1"].value = time.strftime('%Y-%m-%d', time.localtime())
#     sheet["A2"].value = "工号"
#     sheet["B2"].value = "姓名"
#     sheet["C2"].value = "打卡时间"
#     return workbook
#
#
# def add_excel_row(excel_file, info):
#     '''主要逻辑实现'''
#     workbook = openpyxl.load_workbook(excel_file)  # 先打开已存在的表
#     sheet = workbook['Sheet1']
#     row = sheet.max_row + 1
#     print(row)
#
#     sheet.cell(row=row, column=1, value=info.get('工号'))
#     sheet.cell(row=row, column=2, value=info.get('姓名'))
#     # sheet.cell(row=row, column=3, value=info.get(''))
#     sheet.cell(row=row, column=4, value=info.get('部门名称'))
#     sheet.cell(row=row, column=5, value=info.get('岗位名称'))
#     workbook.save(excel_file)
