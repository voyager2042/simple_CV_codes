import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
'''
该程序用于获取hsv的直方图，在hsv图片框里点击可以获取整张图片的直方图，在img图片框里拖拽可以获取拖拽的矩形框（没写可视化）的直方图
'''


def getpic(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        h = hsv[:,:,0].ravel()
        s = hsv[:,:,1].ravel()
        v = hsv[:,:,2].ravel()
        plt.hist(h,255,[0,255],facecolor = 'r')
        plt.hist(s,255,[0,255],facecolor = 'g')
        plt.hist(v,255,[0,255],facecolor = 'b')
        plt.show()
#(0,0,60)(255,90,150)130,200,220   190,240,255
#20,90  100,150  150,255
#60,150

def drag(event,x,y,flags,param):
    global x0,x1,y1,y0
    if event == cv.EVENT_LBUTTONDOWN:
        x0 = x
        y0 = y
    if event == cv.EVENT_LBUTTONUP:
        x1 = x
        y1 = y
        cv.rectangle(img,(x0,y0),(x1,y1),color=(128,0,255),thickness=1)
        imcopy = img.copy()
        imcopy = cv.cvtColor(imcopy,cv.COLOR_BGR2HSV_FULL)
        imcopy = imcopy[y0:y1,x0:x1]
        h = imcopy[:, :, 0].ravel()
        s = imcopy[:, :, 1].ravel()
        v = imcopy[:, :, 2].ravel()
        plt.hist(h, 255, [0, 255], facecolor='r')
        plt.hist(s,255,[0,255],facecolor = 'g')
        plt.hist(v, 255, [0, 255], facecolor='b')
        plt.show()


def videoshow(src,name):
    #src = cv.flip(src, 1)
    cv.imshow(name,src)
    cv.waitKey(25)
    if cv.waitKey(1) & 0xFF == ord('q'):  # 重要，按下q退出
        video.release()  # 释放内存
        cv.destroyAllWindows()
x0 = 0
y0 = 0
x1 = 0
y1 = 0
if __name__ == '__main__':
    video = cv.VideoCapture(0)

    while video.isOpened():
        ret,img = video.read()
        if(ret):
            hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV_FULL)
            splited = cv.split(hsv)
            videoshow(img,'img')
            videoshow(hsv,'hsv')
            videoshow(splited[0],'h')
            videoshow(splited[1],'s')
            videoshow(splited[2],'v')
            cv.setMouseCallback("hsv",getpic)
            cv.setMouseCallback("img", drag)
