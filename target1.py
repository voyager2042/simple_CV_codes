import cv2 as cv
import numpy as np
import os
'''
**非常重要，大多数程序都包含了这个文件**

含有各种功能重要的函数，随意调用
'''
def videoshow(src,name):#显示图像
    #src = cv.flip(src, 1)
    cv.imshow(name,src)
    cv.waitKey(25)
    if cv.waitKey(1) & 0xFF == ord('q'):  # 重要，按下q退出
        video.release()  # 释放内存
        cv.destroyAllWindows()


def adaptive(bgr,ksize,c):#自适应二值化
    img = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img,(11,11),0)
    img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,ksize,c)

    return img

def inter_rec(locxx,locyy):#获取merge所需的矩形，即可以包括内部所有相交矩形的大矩形
    loc1 = locxx
    loc2 = locyy
    for i in range(0, len(locxx)):
        for j in range(0, len(locxx)):
            if i != j:
                Xmax = max(loc1[i][0], locxx[j][0])
                Ymax = max(loc1[i][1], locxx[j][1])
                M = (Xmax, Ymax)
                Xmin = min(loc2[i][0], locyy[j][0])
                Ymin = min(loc2[i][1], locyy[j][1])
                N = (Xmin, Ymin)
                if M[0] < N[0] and M[1] < N[1]: #判断矩形是否相交
                    loc1x = (min(loc1[i][0], locxx[j][0]), min(loc1[i][1], locxx[j][1]))
                    locly = (max(loc2[i][0], locyy[j][0]), max(loc2[i][1], locyy[j][1]))
                    aa=[loc1[i],loc1[j]]
                    bb=[loc2[i],loc2[j]]
                    loc1 = [loc1x if q in aa else q for q in loc1]
                    loc2 = [locly if w in bb else w for w in loc2]
    return loc1,loc2


def shape_select(contours,shape,peri):#选取多边形及其外接矩形xywh坐标
    '''
    :param contours: 输入的轮廓
    :param shape: 需要的几何体边的数目，99表示边大于4的近似曲线几何体（例如圆）
    :param peri:用于获取外接轮廓，peri越小轮廓越接近实际
    :return:shapes为获取的轮廓，coordinates为xywh
    '''
    shapes = []
    coordinates = []
    for i in range(len(contours)):
        perimeter = cv.arcLength(contours[i],closed=True)
        approx = cv.approxPolyDP(contours[i], peri * perimeter, True)  # 确定闭合外框
        corners = len(approx)
        print(corners)
        x, y, w, h = cv.boundingRect(approx)  # 用于确定边长，圆的话单独用霍夫变换求半径
        if (corners == shape and shape != 99):
            single_coordinate = [x, y, w, h]
            coordinates.append(single_coordinate)
            shapes.append(contours[i])
            #print(coordinates)
        if(shape == 99):
            if(corners >4):
                single_coordinate = [x, y, w, h]
                coordinates.append(single_coordinate)
                shapes.append(contours[i])
                # print(co)
    return shapes,coordinates

def hsv_thresh_merge(imghsv,colormin,colormax):#可以实现merge的色块识别
    imgbgr = cv.cvtColor(imghsv, cv.COLOR_HSV2BGR_FULL)
    rect = np.zeros(2)
    line = cv.getStructuringElement(cv.MORPH_RECT, (7, 7), (-1, -1))
    mask = cv.inRange(imghsv, lowerb=colormin, upperb=colormax)
    mask = cv.erode(mask, line)
    mask = cv.dilate(mask, line)
    # mask = cv.dilate(mask, line)
    # mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,line)
    blank = np.zeros(imghsv.shape[:2], np.uint8)
    blank2 = np.zeros(imghsv.shape[:2], np.uint8)
    videoshow(mask,'mask')
    contours,hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    leftup = []
    rightdown = []
    area_rect = 0
    max_rect = []
    flag = False
    if contours:
        for single in contours:
            perimeter = cv.arcLength(single, closed=True)
            approx = cv.approxPolyDP(single, 0.06 * perimeter, True)  # 确定闭合外框
            x, y, w, h = cv.boundingRect(approx)
            lu = (x,y)
            rd = (x+w,y+h)
            leftup.append(lu)
            rightdown.append(rd)
        finleft,finright = inter_rec(leftup,rightdown)
        for j in range(0, len(finleft)):
            cv.rectangle(imgbgr, finleft[j], finright[j], (255, 0, 0), 2)

        for j in range(0,len(finleft)):
            rect_space = (finright[j][0] - finleft[j][0])*(finright[j][1] - finleft[j][1])
            if rect_space >= area_rect:
                area_rect = rect_space
                max_rect = [finleft[j][0],finleft[j][1],finright[j][0] - finleft[j][0],finright[j][1] - finleft[j][1]]#xywh
        flag = True
    #videoshow(imgbgr,'rect')
    return flag,max_rect


def getmask(imghsv,colormin,colormax):#获取色块掩模
    imgbgr = cv.cvtColor(imghsv,cv.COLOR_HSV2BGR_FULL)
    rect = np.zeros(2)
    line = cv.getStructuringElement(cv.MORPH_RECT,(7,7),(-1,-1))
    mask = cv.inRange(imghsv,lowerb=colormin,upperb=colormax)
    mask = cv.erode(mask,line)
    mask = cv.dilate(mask,line)
    #videoshow(mask,'mask')
    return mask


def hsv_thresh(imghsv,colormin,colormax,coordinate,cont_shape):#最常用的色块识别，cont_shape参数是传入的形状轮廓（contours）
    '''
    该函数可以实现利用轮廓的辅助来完成颜色检测，给定轮廓的情况下可以在只检测到轮廓的一部分颜色的时候就能框选到整个色块，
    一般和上面的多边形
    :param coordinate: 未使用
    :param cont_shape: 传入的形状轮廓，
    :return:最终图像掩模、最大所需轮廓、是否检测到、调试用的图片（无实际作用）、最终的xywh坐标
    '''
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
    videoshow(mask,'color')
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
    videoshow(result_max,'max')
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
                cont_max,heriachy = cv.findContours(result_or, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
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

    return result_or, cont_max, flag, result2,coordinate_fin

def get_split(bgr,color_demand,min0,max0,min1,max1,min2,max2):
    '''
    在需检测的色块和BGR三色非常接近的时候可以不必使用hsv，直接分离BGR色域检测更精确
    :param bgr: bgr图片
    :param color_demand:未使用
    :param min0: b域最小值
    :param max0: b域最大值
    :param min1: g域最小值
    :param max1: g域最大值
    :param min2: r域最小值
    :param max2: r域最大值
    :return: 图像掩模
    '''
    img_split = cv.split(bgr)
    line = cv.getStructuringElement(cv.MORPH_RECT, (7, 7), (-1, -1))
    blank = np.zeros(bgr.shape[:2],np.uint8)
    mask = [blank,blank,blank]
    mask[0] = cv.inRange(img_split[0], lowerb=min0, upperb=max0)
    mask[1] = cv.inRange(img_split[1], lowerb=min1, upperb=max1)
    mask[2] = cv.inRange(img_split[2], lowerb=min2, upperb=max2)
    mask_temp = cv.bitwise_and(mask[0],mask[1])
    mask_final = cv.bitwise_and(mask_temp,mask[2])
    mask_final = cv.erode(mask_final, line)
    mask_final = cv.dilate(mask_final, line)
    # videoshow(mask,'mask')
    return mask_final



def disassemble(bgr,color_demand,coordinate,cont_shape):#0B 1G 2R
    '''
    参数与hsv_thresh类似，改成了分离bgr域的检测方式
    :param bgr:
    :param color_demand: 0B 1G 2R，获取所需色域的掩模
    :param coordinate:
    :param cont_shape:
    :return:
    '''
    imgpile = cv.split(bgr)
    b = imgpile[0]
    g = imgpile[1]
    r = imgpile[2]

    aim = imgpile[color_demand]
    rect = np.zeros(2)
    line = cv.getStructuringElement(cv.MORPH_RECT,(9,9),(-1,-1))
    mask = cv.inRange(aim,lowerb=200,upperb=255)
    mask = cv.erode(mask,line)
    mask = cv.dilate(mask,line)
    mask = cv.dilate(mask, line)
    mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,line)
    blank = np.zeros(bgr.shape[:2], np.uint8)
    blank2 = np.zeros(bgr.shape[:2], np.uint8)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    result2 = cv.drawContours(blank, contours, -1, 255, -1)
    videoshow(result2,'bgr')
    flag = False
    largest = 0
    large_num = 0
    xmax = 0
    ymax = 0  # 需要加检测，如果传出的是默认值0就需要认为无效
    result_max = cv.drawContours(blank2, cont_shape, -1, 255, -1)
    cont_max = cont_shape
    if contours:
        max_cont_primitive = max(contours,key=cv.contourArea)
        perimeter1 = cv.arcLength(max_cont_primitive, closed=True)
        approx1 = cv.approxPolyDP(max_cont_primitive, 0.06 * perimeter1, True)  # 确定闭合外框
        x, y, w, h = cv.boundingRect(approx1)
        #print(xmax,ymax)
                # cont_max = contours[i]
        if (xmax + (w / 2) >= x and xmax + (w / 2) <= x + w and ymax + (h / 2) >= y and ymax + (h / 2) <= y + h):
            cont_max = cont_shape
            flag = True
            #print(cont_max)
            result_or = cv.bitwise_or(result2,result_max)
            cont_max = cv.findContours(result_or, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cont_max = max(cont_max,key=cv.contourArea)
            perimeter = cv.arcLength(cont_max, closed=True)
            approx = cv.approxPolyDP(cont_max, 0.06 * perimeter, True)  # 确定闭合外框
            x1, y1, w1, h1 = cv.boundingRect(approx)
            coordinate_color = (x1,y1,w1,h1)
    #result = cv.drawContours(blank, cont_max, -1, 255, -1)
    return result_or, cont_max, flag, result2,coordinate_color

def sampling(hsv,bgr):
    '''
    打印图像中心80*80像素格子里的平均hsv阈值
    :param hsv: hsv图像
    :param bgr:未使用
    :return:
    '''
    sliced_hsv = hsv.copy()
    threshmin = [0,0,0]
    threshmax = [255,255,255]
    h_all = 0
    s_all = 0
    v_all = 0

    '''
    if(xmax and ymax and wmax and hmax):
        #print('captured')
        sliced_hsv = hsv[y:y+h,x:x+w]
        #sliced_hsv = hsv.copy()
        area = sliced_hsv.shape[0]*sliced_hsv.shape[1]
        for lines in sliced_hsv:
            for point in lines:
                #print(point)
                h_all = h_all+point[0]
                s_all = s_all+point[1]
                v_all = v_all+point[2]
        h_all = int(h_all / area)
        s_all = int(s_all / area)
        v_all = int(v_all / area)
        threshmin = [h_all-25,s_all-25,v_all-25]
        threshmax = [h_all+25,s_all+25,v_all+25]
    '''
    center = (int(hsv.shape[1]/2),int(hsv.shape[0]/2))
    sliced_hsv = hsv[center[1]-40:center[1]+40,center[0]-40:center[0]+40]
    hsv = cv.rectangle(hsv,(center[0]-40,center[1]-40),(center[0]+40,center[1]+40),(255,0,255),4)
    videoshow(hsv,'hsv')
    area = 80*80
    sliced_split = cv.split(sliced_hsv)
    all = 0
    alls = [0,0,0]
    for line in sliced_split[0]:
        for point in line:
            #print(point)
            h_all = h_all+point
    for line in sliced_split[1]:
        for point in line:
            #print(point)
            s_all = s_all+point
    for line in sliced_split[2]:
        for point in line:
            #print(point)
            v_all = v_all+point
    h_all = int(h_all / area)
    s_all = int(s_all / area)
    v_all = int(v_all / area)
    threshmin = [h_all - 40, s_all - 40, v_all - 40]
    threshmax = [h_all + 40, s_all + 40, v_all + 40]

    return np.array(threshmin),np.array(threshmax)


'''
target1 complete
'''
def regular_geometry(img,whitemin,whitemax,shape):
    '''
    本文件最重要的函数，检测规则多边形的颜色，可以在颜色检测只检测到一部分的时候获取整个几何体
    :param img:原始图像
    :param whitemin:阈值最小值
    :param whitemax:阈值最大值
    :param shape:形状边的数目
    :return:掩模、轮廓、是否检测到、xywh坐标
    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray, (5,5))
    #ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)#****INV需要看情况更换
    #blurbgr = cv.blur(img,(5,5))
    #th3 = adaptive(blurbgr,7,2)
    th3 = cv.Canny(blur,80,200)
    videoshow(th3,'th3')
    contours, hierarchy = cv.findContours(th3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    areamax = 0
    imax = 0
    coormax = [0, 0, 0, 0]
    shapemax = []
    shapes, coordinates = shape_select(contours, shape,0.04)
    if shapes:

        '''
        for i in range(len(shapes)):
            area = cv.contourArea(shapes[i])
            if area > areamax:
                areamax = area
                imax = i
                coormax = coordinates[i]
        shapemax.append(shapes[imax])        
        
        '''
        shape1 = max(shapes,key=cv.contourArea)
        shapemax.append(shape1)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
    result, cont_max, flag, res2,coordinate_fin = hsv_thresh(hsv, whitemin, whitemax, coormax, shapemax)
    return result, cont_max, flag,coordinate_fin


def red_geometry(img,whitemin,whitemax,shape):
    '''
    hsv色域中红色处于V域的0附近和255附近，无法通过一个阈值检测全部，单独开了一个函数，参数与regular_geometry一致
    :param img:
    :param whitemin:
    :param whitemax:
    :param shape:
    :return:
    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray, (5,5))
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)#****INV需要看情况更换
    blurbgr = cv.blur(img,(5,5))
    #th3 = adaptive(blurbgr,7,2)
    #th3 = cv.Canny(blur,80,200)
    videoshow(th3,'th3')
    contours, hierarchy = cv.findContours(th3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blank2 = np.zeros(img.shape[:2], np.uint8)
    areamax = 0
    imax = 0
    coormax = [0, 0, 0, 0]
    shapemax = []
    shapes, coordinates = shape_select(contours, shape,0.06)
    if shapes:
        for i in range(len(shapes)):
            area = cv.contourArea(shapes[i])
            if area > areamax:
                areamax = area
                imax = i
                coormax = coordinates[i]
        shapemax.append(shapes[imax])
    #print('cor', coormax)
    result3 = cv.drawContours(blank2, shapemax, -1, 255, -1)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
    result, cont_max, flag, res2,coordinate_fin = disassemble(hsv, whitemin, whitemax, coormax, shapemax)
    return result, cont_max, flag,coordinate_fin




mode = 3
whitemin = np.array([110,120,70])
whitemax = np.array([170,255,130])
#whitemin = np.array([0,100,200])#红色H域0，255打死，因为H域是一个环形，红色位于0左右，无法用负数表示的情况下不能处理
#whitemax = np.array([255,255,255])#其实是红色
bluemin = np.array([110,110,10])
bluemax = np.array([200,200,100])

if __name__ == '__main__':

    video = cv.VideoCapture(0)
    while video.isOpened():
        ret,img = video.read()
        if(mode == 0):  #deprecated
            img_adap = adaptive(img,11,3)
            contours,hierarchy = cv.findContours(img_adap,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
            blank = np.zeros(img.shape[:2], np.uint8)
            result = cv.drawContours(blank, contours, -1, 255, -1)
            videoshow(img_adap,'test2')
            videoshow(result,'test')

        elif(mode == 1):
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (11, 11), 0)
            ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            #th3 = adaptive(img,11,4)
            contours, hierarchy = cv.findContours(th3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            blank2 = np.zeros(img.shape[:2], np.uint8)
            #result2 = cv.drawContours(blank2, contours, -1, 255, -1)
            shapes,coordinates = shape_select(contours,4)
            #result3 = cv.drawContours(blank2, shapes, -1, 255, -1)
            areamax = 0
            imax = 0
            coormax = [0,0,0,0]
            shapemax = []
            if shapes:
                for i in range(len(shapes)):
                    area = cv.contourArea(shapes[i])
                    if area > areamax:
                        areamax = area
                        imax = i
                        coormax = coordinates[i]
                shapemax.append(shapes[imax])
                #print('cor',coormax)

            result3 = cv.drawContours(blank2, shapemax, -1, 255, -1)

            hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV_FULL)
            #result,cont_max,flag,res2 = disassemble(img,2,coormax,shapemax)
            result, cont_max, flag, res2,coord = hsv_thresh(hsv,whitemin,whitemax,coormax,shapemax)
            print(flag)
            videoshow(th3, 'test2')
            videoshow(result, 'test')
            videoshow(res2, 'test3')
            videoshow(result3,'test4')

        elif(mode == 2):
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (11, 11), 0)
            ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # th3 = adaptive(img,11,4)
            contours, hierarchy = cv.findContours(th3, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            blank2 = np.zeros(img.shape[:2], np.uint8)
            # result2 = cv.drawContours(blank2, contours, -1, 255, -1)
            shapes, coordinates = shape_select(contours, 4)
            result3 = cv.drawContours(blank2, shapes, -1, 255, -1)
            areamax = 0
            imax = 0
            coormax = [0, 0, 0, 0]
            shapemax = []
            if shapes:
                for i in range(len(shapes)):
                    area = cv.contourArea(contours[i])
                    if area > areamax:
                        areamax = area
                        imax = i
                        coormax = coordinates[i]
                        shapemax = shapes[i]
                #print('cor', coormax)

            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
            # result,cont_max,flag,res2 = disassemble(img,2,coormax,shapemax)
            result, cont_max, flag, res2 = hsv_thresh(hsv, whitemin, whitemax, coormax, shapemax)
        elif(mode == 3):
            img2 = img.copy()
            result, cont_max, flag,coordinate_fin = regular_geometry(img,bluemin,bluemax,4)
            print(flag)
            videoshow(result,'result')
            if(isinstance(cont_max, np.ndarray)):
                img2 = cv.drawContours(img2,cont_max,-1,(255,255,255),4)
            if(coordinate_fin):
                print(coordinate_fin)
                img2 = cv.rectangle(img2,(coordinate_fin[0],coordinate_fin[1]),(coordinate_fin[0]+coordinate_fin[2],coordinate_fin[1]+coordinate_fin[3]),(255,255,0),2)
            videoshow(img2,'img')

