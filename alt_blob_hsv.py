import cv2 as cv
import numpy as np
'''
该程序用于预处理图像、需要通过其他程序获取阈值、利用阈值对色块进行threshold（色块识别），腐蚀或膨胀腐蚀
这套预处理和色块识别的方式亲测抗干扰性能较强
'''
def hsv_pre(imgBGR):
    hsv = cv.cvtColor(imgBGR,cv.COLOR_BGR2HSV_FULL)
    hsv = cv.blur(hsv,(5,5))
    return hsv

def morphological_operation(frame):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 获取图像结构化元素
    dst = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)  # 闭操作
    return dst

def hsv_thresh(imghsv,colormin,colormax):
    rect = np.zeros(2)
    line = cv.getStructuringElement(cv.MORPH_RECT,(9,9),(-1,-1))
    mask = cv.inRange(imghsv,colormin,colormax)
    mask = cv.erode(mask,line)
    mask = cv.dilate(mask,line)
    #mask = cv.dilate(mask, line)
    #mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,line)
    blank = np.zeros(imghsv.shape[:2],np.uint8)

    contours,hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    result = cv.drawContours(blank,contours,-1,255,-1)
    return result

def videoshow(src,name):
    #src = cv.flip(src, 1)
    cv.imshow(name,src)
    cv.waitKey(25)
    if cv.waitKey(1) & 0xFF == ord('q'):  # 重要，按下q退出
        video.release()  # 释放内存
        cv.destroyAllWindows()

blackmin = (119,98,87)
blackmax = (133,159,147)

if __name__ == '__main__':
    video = cv.VideoCapture(0)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 224)
    video.set(cv.CAP_PROP_FRAME_WIDTH, 224)
    while video.isOpened():
        ret,img = video.read()
        if(ret):
            hsv = hsv_pre(img)
            result = hsv_thresh(hsv,blackmin,blackmax)
            videoshow(result,'test')


