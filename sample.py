import target1
import cv2 as cv
'''
target1中的sample函数应用
'''
if __name__ == '__main__':
    video = cv.VideoCapture(0)

    while video.isOpened():
        ret, img = video.read()
        hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV_FULL)
        threshmin,threshmax = target1.sampling(hsv,img)
        print(threshmin,threshmax)
        target1.videoshow(img,'img')





