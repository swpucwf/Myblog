import math

import cv2.cv2 as cv2
import numpy as np

def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # 圆型检测
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,param1=200, param2=30, minRadius=200, maxRadius=300)
    img_copy = img.copy()

    isNG = False

    if circles is None:
        print("找圆失败！")

    else:
        # 找出圆型
        circles = np.uint16(np.around(circles))
        a, b, c = circles.shape
        print("circle:",circles.shape)
        print(circles)
        for i in range(b):
            # 外圆轮廓
            cv2.circle(img_copy, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
            # 绿色圆心
            cv2.circle(img_copy, (circles[0][i][0], circles[0][i][1]), radius=2, color=(0, 255, 0), thickness=3,
                       lineType=cv2.LINE_AA)  # draw center of circle

        # cv2.imshow("findCircle", img_copy)
        # cv2.waitKey(0)
            # 极坐标变换
            # x = c_x -r ,y = c_y-r ,w= h = 2r , C = 2*pi*R
            x = circles[0][i][0] - circles[0][i][2]
            y = circles[0][i][1] - circles[0][i][2]
            w = h = 2 * circles[0][i][2]
            center = (circles[0][i][0], circles[0][i][1])
            radius = circles[0][i][2]
            C = 2 * math.pi * radius
            print(C,radius)

            ROI = img[y:y+h,x:x+w].copy()
            cv2.imshow('ROI',ROI)
            trans_center = (center[0]-x, center[1]-y)
            polarImg = cv2.warpPolar(ROI,(int(radius),int(C)),trans_center,radius,cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)

            # 镜像
            polarImg = cv2.flip(polarImg,1) #镜像
            # 转置
            polarImg = cv2.transpose(polarImg)#转置
            cv2.imwrite('polarImg.png',polarImg)

            result = reader.readtext(polarImg)
            print(result)
            if len(result) > 0:
                for i in range(0, len(result)):
                    print(result[i][1])
                    if (result[i][2] < 0.4):
                        continue
                    for j in range(4):
                        if j > 0:
                            cv2.line(polarImg, (tuple(result[i][0][j - 1])), (tuple(result[i][0][j])), (0, 255, 0), 2,
                                     cv2.LINE_AA)
                    cv2.line(polarImg, (tuple(result[i][0][0])), (tuple(result[i][0][3])), (0, 255, 0), 2, cv2.LINE_AA)
                    strText = result[i][1].replace(' ', '')
                    cv2.putText(polarImg, strText, (result[i][0][3][0], result[i][0][3][1] + 20), 0, 0.8, (0, 0, 255),
                                2)

                cv2.imshow('polarImg-OCR', polarImg)
            polarImg = cv2.flip(polarImg, 0)  # 镜像
            polarImg = cv2.transpose(polarImg)  # 转置
            polarImg_Inv = cv2.warpPolar(polarImg, (w, h), trans_center, radius, cv2.INTER_LINEAR + \
                                         cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)
            cv2.imshow('polarImg_Inv', polarImg_Inv)


            mask = np.zeros((h,w,1),np.uint8)
            cv2.circle(mask,trans_center,radius-3,(255,255,255),-1, cv2.LINE_AA)
            cv2.imshow('mask', mask)
            ROI = img[y:y+h,x:x+w]
            for i in range(0,ROI.shape[0]):
                for j in range(0, ROI.shape[1]):
                  if mask[i,j] > 0:
                    ROI[i,j] = polarImg_Inv[i,j]
            cv2.imshow('result', img)
            cv2.waitKey(0)
if __name__ == '__main__':
    img = cv2.imread("test.png")
    detect(img)
