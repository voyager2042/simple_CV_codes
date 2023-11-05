
import cv2 as cv
import numpy as np
import target1

'''
没改函数名字，但这个真的是调整识别线段函数的参数用的
'''
def detect_circle(src,x1,x2,y1,y2):
    rho = cv.getTrackbarPos('rho', 'color_adjust1')
    theta = cv.getTrackbarPos('theta', 'color_adjust1')
    threshold = cv.getTrackbarPos('threshold', 'color_adjust1')
    lines = cv.getTrackbarPos('lines', 'color_adjust1')
    minLineLength = cv.getTrackbarPos('minLineLength', 'color_adjust1')
    maxLineGap = cv.getTrackbarPos('maxLineGap', 'color_adjust1')
    if rho == 0:
        rho = 1
    if theta == 0:
        theta = 1
    if threshold == 0:
        threshold = 1
    if lines == 0:
        lines = 1
    if minLineLength == 0:
        minLineLength = 1
    if maxLineGap == 0:
        maxLineGap = 1
    src1 = src.copy()
    src1 = np.uint8(src1)
    kernel = np.ones((3, 3), np.uint8)
    src1 = cv.morphologyEx(src1, cv.MORPH_OPEN, kernel)
    #src = cv.GaussianBlur(src, (7, 7), 0)
    src1 = cv.Canny(src1, 100,250)
    if (x1 and x2 and y1 and y2):
        src1 = src1[y1:y2, x1:x2]

    point = (0,0)
    k = 0
    linepoint = []
    lines = cv.HoughLinesP(src1,rho,np.pi/theta,threshold,minLineLength=minLineLength,maxLineGap=maxLineGap)
    lengthmax = 0
    cirxy = (0, 0)
    if (isinstance(lines, np.ndarray)):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if(abs(x2-x1) == 0):
                x2 = x1+2
            k = int((y2-y1)/(x2-x1))
            x1L = 0
            y1L = k * (0 - x1) + y1
            x2R = 640
            y2R = k * (640 - x1) + y1
            cv.line(src, (x1, y1), (x2, y2), (0, 255, 0), 2)
            linepoint.append([x1,y1])
            linepoint.append([x2,y2])
    return lines

def nothing(x):
    pass

def main():
    cv.namedWindow("color_adjust1")
    cv.resizeWindow("color_adjust1",240,360)
    cv.createTrackbar("rho", "color_adjust1", 1, 255, nothing)
    cv.createTrackbar("theta", "color_adjust1", 1, 360, nothing)
    cv.createTrackbar("threshold", "color_adjust1", 1, 255, nothing)
    cv.createTrackbar("lines", "color_adjust1", 1, 255, nothing)
    cv.createTrackbar("minLineLength", "color_adjust1", 1, 255, nothing)
    cv.createTrackbar("maxLineGap", "color_adjust1", 1, 255, nothing)

    capture = cv.VideoCapture(0)  # 打开电脑自带摄像头，
    #capture.set(44, 0)  # disable wb

    while True:
        ret, frame = capture.read()
        if(ret):
            lines = detect_circle(frame,0,0,0,0)
            #frame2 = frame.copy()
            #frame2 = cv.circle(frame2,center,length,(255,0,255),3)
            target1.videoshow(frame,'frame')

if __name__ == '__main__':
    main()