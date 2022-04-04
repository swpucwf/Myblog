import numpy as np
import cv2.cv2 as cv2

if __name__ == '__main__':

    font = cv2.FONT_HERSHEY_SIMPLEX

    img = cv2.imread('./test02.png')
    cv2.imshow('src', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k1 = np.zeros((7, 7), np.uint8)
    pts = np.array([[2, 0], [4, 0], [6, 2], [6, 4], [4, 6], [2, 6], [0, 4], [0, 2]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(k1, [pts], (1, 1, 1), cv2.LINE_AA)
    k1[5, 1] = 1
    k1[6, 2:5] = 1

    print(k1)

    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k1)
    cv2.imshow('opening', opening)

    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k1)
    cv2.imshow('closing', closing)

    diff = cv2.absdiff(opening, closing)
    cv2.imshow('diff', diff)

    ret, thresh = cv2.threshold(diff, 80, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    isNG = False

    if len(contours) > 0:
        isNG = True
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    if isNG:
        rect, basline = cv2.getTextSize('Detect NG', font, 1.0, 2)
        cv2.rectangle(img, (10, 10, int(rect[0] * 0.7), rect[1]), (212, 233, 252), -1, 8)
        cv2.putText(img, 'Detect NG', (10, 5 + rect[1]), font, 0.7, (0, 0, 255), 2)
    else:
        rect, basline = cv2.getTextSize('Detect OK', font, 1.0, 2)
        cv2.rectangle(img, (10, 10, int(rect[0] * 0.7), rect[1]), (212, 233, 252), -1, 8)
        cv2.putText(img, 'Detect OK', (10, 5 + rect[1]), font, 0.7, (0, 200, 0), 2)
    cv2.imshow('meshDefects', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()