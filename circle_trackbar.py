import cv2 as cv
import numpy as np
import target1
import pyrealsense2

'''
用于调整识别圆函数里的参数，加了可视化
'''
def detect_circle(src,x1,x2,y1,y2):
    dp = cv.getTrackbarPos('dp', 'color_adjust1')
    mindist = cv.getTrackbarPos('mindist', 'color_adjust1')
    param1 = cv.getTrackbarPos('param1', 'color_adjust1')
    param2 = cv.getTrackbarPos('param2', 'color_adjust1')
    minradius = cv.getTrackbarPos('minradius', 'color_adjust1')
    if dp == 0:
        dp = 1
    if mindist == 0:
        mindist = 1
    if param1 == 0:
        param1 = 1
    if param2 == 0:
        param2 = 1
    if minradius == 0:
        minradius = 1
    src1 = src.copy()
    src1 = np.uint8(src1)
    kernel = np.ones((3, 3), np.uint8)
    src1 = cv.morphologyEx(src1, cv.MORPH_OPEN, kernel)
    #src = cv.GaussianBlur(src, (7, 7), 0)
    src1 = cv.Canny(src1, 100,250)
    if (x1 and x2 and y1 and y2):
        src1 = src1[y1:y2, x1:x2]


    circles = cv.HoughCircles(src1, method=cv.HOUGH_GRADIENT, dp=dp, minDist=mindist, param1=param1, param2=param2,minRadius = minradius)
    lengthmax = 0
    cirxy = (0, 0)
    if (isinstance(circles, np.ndarray)):
        circles = np.uint16(np.around(circles))
        for cir in circles[0]:

            shape = 'circle'
            length = cir[2]
            if length > lengthmax:
                lengthmax = length
                cirxy = (cir[0], cir[1])
    return cirxy,lengthmax

def nothing(x):
    pass

def main():
    cv.namedWindow("color_adjust1")
    cv.resizeWindow("color_adjust1",240,360)
    cv.createTrackbar("dp", "color_adjust1", 2, 5, nothing)
    cv.createTrackbar("mindist", "color_adjust1", 1, 255, nothing)
    cv.createTrackbar("param1", "color_adjust1", 1, 255, nothing)
    cv.createTrackbar("param2", "color_adjust1", 1, 255, nothing)
    cv.createTrackbar("minradius", "color_adjust1", 1, 255, nothing)

    capture = cv.VideoCapture(0)  # 打开电脑自带摄像头，
    #capture.set(44, 0)  # disable wb

    while True:
        ret, frame = capture.read()
        if(ret):
            center,length = detect_circle(frame,0,0,0,0)
            frame2 = frame.copy()
            frame2 = cv.circle(frame2,center,length,(255,0,255),3)
            target1.videoshow(frame2,'frame')

if __name__ == '__main__':
    main()