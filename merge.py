import cv2
import numpy as np
'''
实现了openmv的merge功能
'''


def inter_rec(locxx,locyy):
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

if __name__ == '__main__':
    # 矩形左上角坐标：
    locxx = [(478, 528), (185, 525), (423, 489), (200, 474), (595, 467), (488, 467), (313, 454), (391, 442), ( 244, 435), (240, 418), (431, 404), (437, 403), (352, 403), (365, 343), (303, 338), (436, 331), (343, 331), (222, 318), (353, 258), (241, 163)]
    #矩形右下角坐标：
    locyy = [(494, 552), (201, 556), (485, 509), (222, 524), (613, 495), (544, 511), (340, 481), (423, 507), (308, 490), (264, 475), (468, 471), (459, 447), (385, 457), (382, 370), (335, 373), (459, 350), (365, 362), (294, 485), (391, 298), (542, 429)]

    Img = np.zeros([670, 700, 3], np.uint8) + 255
    Img1 = Img.copy()
    finx, finy = inter_rec(locxx, locyy)
    for i in range(0, len(locxx)):
        cv2.rectangle(Img, locxx[i], locyy[i], (0, 0, 255), 2)
        cv2.imshow("img", Img)
    for j in range(0, len(finx)):
        cv2.rectangle(Img1, finx[j], finy[j], (255, 0, 0), 2)
    cv2.imshow("result", Img1)
    cv2.waitKey(0)
