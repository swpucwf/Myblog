#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/12/17 19:19
# @Author  : 陈伟峰
# @Site    : 
# @File    : test01.py
# @Software: PyCharm
import cv2
def show_img():
    img =cv2.imread("../images/img.png")
    print(type(img))
    print("图像形状:",img.shape)
    # 图像宽高 image.shape
    print(img.shape)
    # 图像深度 image
    # 图像数据类型 image.dtype
    print(img.dtype)
    # 图像通道 image.shape
    # 如何加载不同通道的图像
    cv2.imshow("lisa",img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    show_img()