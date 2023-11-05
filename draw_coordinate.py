import cv2 as cv
import numpy as np
import math
#也可以快于一秒一次发送数据，一秒显示一次即可

'''
2023年电赛G题中有连接各点绘制路径的要求，该程序可以实现
'''
def display_coordinate(img,imgw,imgh,x,y):
    img_dis = np.zeros(img.shape[:2],np.uint8)
    img_dis = cv.putText(img_dis,str(x)+','+str(y),(int(imgw / 2),imgh - 20),cv.FONT_HERSHEY_PLAIN,color = 255)
    return img_dis

def draw_trail(img_no_copy,dots):
    flag_num = 0
    sum_trail = 0
    for i in range(len(dots)):
        if(i == 0):
            flag_num = flag_num +1
            pass
        else:
            img_no_copy = cv.line(img_no_copy,dots[i-1],dots[i],255,3)
            length = math.sqrt(((dots[i-1][0] - dots[i][0]) ** 2) + (dots[i-1][1] - dots[i][1]) ** 2)
            sum_trail = sum_trail + length
    img_no_copy = cv.putText(img_no_copy, str(int(sum_trail))+'pixels', (int(img_no_copy.shape[1] / 2), img_no_copy.shape[0] - 20), cv.FONT_HERSHEY_PLAIN,1,
                         color=255)
    return img_no_copy,sum_trail



dots = ((1,1),(20,13),(22,35),(40,50),(68,79),(80,180))
IMGX = 224
IMGY = 224
if __name__ == '__main__':
    img = np.zeros((IMGX,IMGY),np.uint8)
    _,sum = draw_trail(img,dots)
    cv.imshow('img',img)
    print(sum)
    cv.waitKey()