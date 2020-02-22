import cv2
import time
import cv2
import numpy as np
import math
import sys

CONTROL_THRESH1 = 'Thresh1'
CONTROL_THRESH2 = 'Thresh2'
CONTROL_LINE_WIDTH = 'Line Width'

THRESH_VALUE1 = 50
THRESH_VALUE2 = 1


def threshValue1(value1):
    global THRESH_VALUE1
    THRESH_VALUE1 = value1
    return


def threshValue2(value1):
    global THRESH_VALUE2
    THRESH_VALUE2 = value1
    return


def lineLength(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return int(round(math.sqrt(dx * dx + dy * dy)))

cam = cv2.VideoCapture(0)

cv2.namedWindow("source")
cv2.namedWindow("thresh")
cv2.createTrackbar(CONTROL_THRESH1, "thresh", THRESH_VALUE1, 255, threshValue1)
cv2.createTrackbar(CONTROL_THRESH2, "thresh", THRESH_VALUE2, 255, threshValue2)

# Write some Text
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

while True:
    ret, frame = cam.read()
    if not ret:
        exit()
        break
    image = frame

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    retval, thresh_image = cv2.threshold(
        gray_image, THRESH_VALUE1, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    thresh_image = cv2.erode(thresh_image, kernel, 7)
    thresh_image = cv2.dilate(thresh_image, kernel, 7)

    retval, thresh_image = cv2.threshold(
        thresh_image, THRESH_VALUE2, 255, cv2.THRESH_BINARY)

    img, contours, hierarchy = cv2.findContours(
        thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    if not contours:
        print("no objects found")
    else:
        c = contours[0]

        x, y, w, h = cv2.boundingRect(c)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        MIDDLE_X, MIDDLE_Y = image.shape[1]//2, image.shape[0]//2
        # print(x, y, x+w, y+h)
        # print(MIDDLE_X, MIDDLE_Y)
        cv2.circle(image, (MIDDLE_X, MIDDLE_Y), 2, (0, 0, 255), -1)
        cv2.circle(image, (x+w//2, y+h//2), 2, (0, 255, 255), -1)
        cv2.line(image, (MIDDLE_X, MIDDLE_Y),
                 (x+w//2, y+h//2), (255, 0, 0), 1, 8)

        d = (MIDDLE_X-(x+w//2))
        # print("distance :", d)

        k = cv2.waitKey(1)
        if k % 256 == 32:
            print("space pressed!")
            cv2.putText(image, "space pressed", bottomLeftCornerOfText,
                    font, fontScale, fontColor, lineType)
            cv2.imshow("marking contours image", image)
            cv2.waitKey(5000)
        else:
            cv2.putText(image, "distance: "+str(d), bottomLeftCornerOfText,
                    font, fontScale, fontColor, lineType)
            cv2.imshow("marking contours image", image)

cam.release()
cv2.destroyAllWindows()