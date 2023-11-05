import cv2 as cv
import numpy as np
import target1
'''
最常用的选取hsv阈值程序，点击h图片框还可以获取该点的hsv值
'''
def nothing(x):
    pass


def morphological_operation(frame):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 获取图像结构化元素
    dst = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)  # 闭操作
    return dst
def hsv_thresh(imghsv,colormin,colormax,coordinate,cont_shape):
    imgbgr = cv.cvtColor(imghsv,cv.COLOR_HSV2BGR_FULL)
    rect = np.zeros(2)
    line = cv.getStructuringElement(cv.MORPH_RECT,(7,7),(-1,-1))
    mask = cv.inRange(imghsv,lowerb=colormin,upperb=colormax)
    #mask = cv.erode(mask,line)
    #mask = cv.dilate(mask,line)
    #mask = cv.dilate(mask, line)
    #mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,line)
    blank = np.zeros(imghsv.shape[:2],np.uint8)
    blank2 = np.zeros(imghsv.shape[:2], np.uint8)
    blank3 = np.zeros(imghsv.shape[:2], np.uint8)
    #videoshow(mask,'mask')
    contours,hierarchy = cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
    result2 = cv.drawContours(blank, contours, -1, 255, -1)
    target1.videoshow(mask,'color')
    flag = False
    largest = 0
    large_num = 0
    xmax = 0
    ymax = 0#需要加检测，如果传出的是默认值0就需要认为无效
    wmax = 0
    hmax = 0
    result_max = cv.drawContours(blank2, cont_shape, -1, 255, -1)
    xfin = 0
    yfin = 0
    wfin = 0
    hfin = 0
    x_cont=y_cont=w_cont=h_cont=0
    coordinate_fin = []
    #result_max = cv.erode(mask,line)
    #result_max = cv.dilate(mask,line)
    target1.videoshow(result_max,'max')
    result_or = result_max
    cont_max = []
    if cont_shape:
        cont_max = max(cont_shape,key=cv.contourArea)
        for i in cont_shape:
            perimeter1 = cv.arcLength(i, closed=True)
            approx1 = cv.approxPolyDP(i, 0.06 * perimeter1, True)  # 确定闭合外框
            x_cont, y_cont, w_cont, h_cont = cv.boundingRect(approx1)
    if contours:
        cont_max = max(contours,key=cv.contourArea)
        perimeter = cv.arcLength(cont_max, closed=True)
        approx = cv.approxPolyDP(cont_max, 0.06 * perimeter, True)  # 确定闭合外框
        x, y, w, h = cv.boundingRect(approx)
        xmax = x
        ymax = y
        wmax = w
        hmax = h
        if(cont_shape):
            if (xmax + (wmax / 2) >= x_cont-25 and xmax + (wmax / 2) <= x_cont + w_cont+25 and ymax + (hmax / 2) >= y_cont-25 and ymax + (hmax / 2) <= y_cont + h_cont+25):
                #cont_max = cont_shape
                flag = True
                #print(cont_max)
                result_or = cv.bitwise_or(result2, result_max)
                cont_max,heriachy = cv.findContours(result_or, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                cont_max = max(cont_max,key=cv.contourArea)
                perimeter = cv.arcLength(cont_max, closed=True)
                approx = cv.approxPolyDP(cont_max, 0.06 * perimeter, True)  # 确定闭合外框
                x,y,w,h = cv.boundingRect(approx)
                coordinate_fin = (x,y,w,h)
        else:
            flag = True
            coordinate_fin = (xmax,ymax,wmax,hmax)
            if(isinstance(cont_max, np.ndarray)):
                #cont_max = max(cont_max, key=cv.contourArea)
                result_or = cv.drawContours(blank3,cont_max,-1,255,-1)

    return result_or

def color_detetc(frame):

    hmin1 = cv.getTrackbarPos('hmin1', 'color_adjust1')
    hmax1 = cv.getTrackbarPos('hmax1', 'color_adjust1')
    smin1 = cv.getTrackbarPos('smin1', 'color_adjust1')
    smax1 = cv.getTrackbarPos('smax1', 'color_adjust1')
    vmin1 = cv.getTrackbarPos('vmin1', 'color_adjust1')
    vmax1 = cv.getTrackbarPos('vmax1', 'color_adjust1')

    hmin2 = cv.getTrackbarPos('hmin2', 'color_adjust2')
    hmax2 = cv.getTrackbarPos('hmax2', 'color_adjust2')
    smin2 = cv.getTrackbarPos('smin2', 'color_adjust2')
    smax2 = cv.getTrackbarPos('smax2', 'color_adjust2')
    vmin2 = cv.getTrackbarPos('vmin2', 'color_adjust2')
    vmax2 = cv.getTrackbarPos('vmax2', 'color_adjust2')
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)  # hsv 色彩空间 分割肤色
    lower_hsv1 = np.array([hmin1, smin1, vmin1])
    upper_hsv1 = np.array([hmax1, smax1, vmax1])
    mask1 = cv.inRange(hsv, lowerb=lower_hsv1, upperb=upper_hsv1)  # hsv 掩码
    lower_hsv2 = np.array([hmin2, smin2, vmin2])
    upper_hsv2 = np.array([hmax2, smax2, vmax2])
    mask2 = hsv_thresh(hsv,lower_hsv2,upper_hsv2,None,None)  # hsv 掩码
    ret, thresh1 = cv.threshold(mask1, 40, 255, cv.THRESH_BINARY)  # 二值化处理
    ret, thresh2 = cv.threshold(mask2, 40, 255, cv.THRESH_BINARY)  # 二值化处理

    return thresh1, thresh2


#(0,0,0)(100,40,80)(50,50,75   190.50.90  190.50.90
#(0,0,0)(50,50,95)(,,120),,100,,100,,85,,85


#min(111,0,2)max(255,111,82)L[133 159 147] [119  98  87]
#min(0,0,0)max(255,234,96)T
def main():
    def getpos(event, x, y, flags, param):

        if event == cv.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
            print(h[y, x])
    cv.namedWindow("color_adjust1")
    cv.namedWindow("color_adjust2")
    cv.createTrackbar("hmin1", "color_adjust1", 16, 255, nothing)
    cv.createTrackbar("hmax1", "color_adjust1", 31, 255, nothing)
    cv.createTrackbar("smin1", "color_adjust1", 119, 255, nothing)
    cv.createTrackbar("smax1", "color_adjust1", 255, 255, nothing)
    cv.createTrackbar("vmin1", "color_adjust1", 0, 255, nothing)
    cv.createTrackbar("vmax1", "color_adjust1", 255, 255, nothing)

    cv.createTrackbar("hmin2", "color_adjust2", 130, 255, nothing)
    cv.createTrackbar("hmax2", "color_adjust2", 190, 255, nothing)
    cv.createTrackbar("smin2", "color_adjust2", 200, 255, nothing)
    cv.createTrackbar("smax2", "color_adjust2", 240, 255, nothing)
    cv.createTrackbar("vmin2", "color_adjust2", 220, 255, nothing)
    cv.createTrackbar("vmax2", "color_adjust2", 255, 255, nothing)
    capture = cv.VideoCapture(0)  # 打开电脑自带摄像头，
    #capture.set(44, 0)  # disable wb

    while True:
        ret, frame = capture.read()
        if(ret):

            mask1, mask2 = color_detetc(frame)
            scr1 = morphological_operation(mask1)
            scr2 = morphological_operation(mask2)
            h = cv.cvtColor(frame,cv.COLOR_BGR2HSV_FULL)

            contours1, heriachy1 = cv.findContours(scr1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 获取轮廓点集(坐标)

            contours2, heriachy2 = cv.findContours(scr2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 获取轮廓点集(坐标)
            cv.drawContours(frame, contours1, -1, (0, 0, 255), 2)
            cv.drawContours(frame, contours2, -1, (0, 255, 0), 2)
            for i, contour in enumerate(contours1):
                area1 = cv.contourArea(contour)
                if area1 > 20:
                    (x1, y1), radius1 = cv.minEnclosingCircle(contours1[i])
                    x1 = int(x1)
                    y1 = int(y1)
                    center1 = (int(x1), int(y1))
                    radius1 = int(radius1)
                    cv.circle(frame, center1, 3, (0, 0, 255), -1)  # 画出重心
                    #print("黄色坐标:", (x1, y1))
                    cv.putText(frame, "yellow:", (x1, y1), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, [255, 255, 255])
            for k, contour in enumerate(contours2):
                area2 = cv.contourArea(contour)
                if area2 > 20:
                    (x2, y2), radius2 = cv.minEnclosingCircle(contours2[k])
                    x2 = int(x2)
                    y2 = int(y2)
                    center2 = (int(x2), int(y2))
                    radius2 = int(radius2)
                    cv.circle(frame, center2, 3, (0, 0, 255), -1)  # 画出重心
                    #print("蓝色坐标:", (x2, y2))
                    cv.putText(frame, "blue:", (x2, y2), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, [255, 255, 255])
            cv.imshow("mask1", mask1)
            cv.imshow("mask2", mask2)
            cv.imshow("frame", frame)
            cv.imshow('h',h)
            cv.setMouseCallback("h", getpos)
            c = cv.waitKey(50)
            if c == 27:
                break


main()

cv.waitKey(0)
cv.destroyAllWindows()

