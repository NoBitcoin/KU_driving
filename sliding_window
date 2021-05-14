# perspective transformation, sliding window
# practice python file

import cv2
import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

cap=cv2.VideoCapture("/home/siyoung/catkin_ws/src/KU_driving/line_detect/nnum/kmu_track.mkv")

right_slope = []
left_slope = []
middle_slope = []

def sliding_window(image):
    
    # img_B, img_G, img_R = cv2.split(image)

    # y = 250
    
    # nonzero_x = img_R[y, :].nonzero()[0]

    x = image.shape[1]
    y = image.shape[0]
    middle_x = x//2
    margin = 30
    nonzero_left_sum = []
    nonzero_right_sum = []


    # nonzero_y = img[:,x].nonzero()[0]   
    # nonzero_sum = sum(nonzero_y)

    for i in range(x):
        # every x
        if( i >= 0 and i < middle_x - margin):
            nonzero_y = image[ : , i ].nonzero()[0]
            nonzero_left = len(nonzero_y)
            nonzero_left_sum.append(nonzero_left)
            # print(nonzero_left_sum)
        elif( i <= x and i > middle_x + margin):
            nonzero_y = image[ : , i ].nonzero()[0]
            nonzero_right = len(nonzero_y)
            nonzero_right_sum.append(nonzero_right)
            # print(nonzero_right_sum)
        else:
            pass
    
    left_PixelSum_Max = max(nonzero_left_sum)
    left_PixelSum_Max_index = nonzero_left_sum.index(left_PixelSum_Max)
    #print(left_PixelSum_Max_index)
    Left_Max = left_PixelSum_Max_index
    # print()
    
    right_PixelSum_Max = max(nonzero_right_sum)
    right_PixelSum_Max_index = nonzero_right_sum.index(right_PixelSum_Max)
    Right_Max = right_PixelSum_Max_index + middle_x + margin
    print(Left_Max, Right_Max)
    cv2.rectangle(image, (Left_Max - 20, y), (Left_Max + 20, y - 40), (0,255,0), 1)
    cv2.rectangle(image, (Right_Max - 20, y), (Right_Max + 20, y - 40), (0,255,0), 1)

    return image

def warpping(image):
    (h,w)=(image.shape[0],image.shape[1])

    source=np.float32([[200,h-165],[w-150,h-160],[w, h-100],[0,h-100]])
    des=np.float32([[0,0],[w,0],[w,h],[0,h]])

    transform_matrix=cv2.getPerspectiveTransform(source,des)
    minv=cv2.getPerspectiveTransform(des,source)
    _image=cv2.warpPerspective(image,transform_matrix,(w,h))

    return _image, minv

def roi(image):
    x=int(image.shape[1])
    y=int(image.shape[0])

    _shape=np.array([[int(0), int(y)],[int(0),int(0.85*y)],
                    [int(0.35*x),int(0.54*y)],
                    [int(0.62*x),int(0.52*y)],
                    [int(x),int(0.7*y)],
                    [int(x),int(y)]])
    mask=np.zeros_like(image)

    if len(image.shape)>2:
        channel_count=image.shape[2]
        ignore_mask_color=(255,)*channel_count
    else:
        ignore_mask_color=255 
        
    cv2.fillPoly(mask,np.int32([_shape]),ignore_mask_color)
    masked_image=cv2.bitwise_and(image,mask)
    #cv2.imshow('mid',masked_image)
    return masked_image


while(True):
    ret,src=cap.read()
    src=cv2.resize(src,(640,360))
    
    dst=cv.Canny(src,50,200,None,3)
    dst=roi(dst)

    cdst=cv.cvtColor(dst,cv.COLOR_GRAY2BGR)
    cdstP=np.copy(cdst)

    cdstP_bird, minverse = warpping(cdstP)
    cdstP_bird = sliding_window(cdstP_bird)
    # rotated_cdstP_bird = cv2.getRotationMatrix2D((cdstP_bird.shape[1]/2, cdstP_bird.shape[0]/2), -15, 1)
    # cdstP_bird = cv2.warpAffine(cdstP_bird, rotated_cdstP_bird,(cdstP_bird.shape[1], cdstP_bird.shape[0]))
    #cv2.rectangle(cdstP_bird, (100, 0), (150, 200), (0,255,0), 3)
    cv2.imshow('test', cdstP_bird)

    linesP=cv.HoughLinesP(dst,1,np.pi/180,30,None,50,10)
    
    if linesP is not None:
        for i in range(0,len(linesP)):
            l=linesP[i][0]
            cv.line(cdstP,(l[0],l[1]),(l[2],l[3]), (0, 0, 255),3,cv.LINE_AA)
            try:
                slope = float(l[1] - l[3]) / float(l[2] - l[0])
            except:
                pass
            # print(slope)
            if(slope < 1 and slope > -1 and slope != 0):
                if(slope < 0):
                    right_slope.append(slope)
                else:
                    left_slope.append(slope)
            else:
                if(slope == 0):
                    pass
                else: 
                    middle_slope.append(slope)

    img_B, img_G, img_R = cv2.split(cdstP)

    y = 250
    
    nonzero_x = img_R[y, :].nonzero()[0]


    mean_right_slope = float(sum(right_slope) / len(right_slope))
    mean_left_slope = float(sum(left_slope) / len(left_slope))
    
    x_1 = (cdstP.shape[0] - 250 + (mean_right_slope*nonzero_x[0])) / mean_right_slope 
    x_2 = ((mean_right_slope*nonzero_x[0]) - 250) / mean_right_slope
    (x_1, cdstP.shape[0]), (x_2, 0)
    
    x_3 = (cdstP.shape[0] - 250 + (mean_left_slope*nonzero_x[-1])) / mean_left_slope 
    x_4 = ((mean_left_slope*nonzero_x[-1]) - 250) / mean_left_slope
    (x_3, cdstP.shape[0]), (x_4, 0)

    cv2.line(cdstP,(int(x_1), cdstP.shape[0]), (int(x_2), 0), (255, 0, 0),3,cv.LINE_AA)
    cv2.line(cdstP,(int(x_3), cdstP.shape[0]), (int(x_4), 0), (255, 0, 0),3,cv.LINE_AA)

    # img_B, img_G, img_R = cv2.split(cdstP)

    # y = 250
    
    # nonzero_x = img_R[y, :].nonzero()[0]

    img_R = cv2.merge((img_R, img_R, img_R))

    if(abs(nonzero_x[0] - nonzero_x[-1]) < 70):
        try:
            cv2.imshow("asdf", img_R)
        except:
            pass

    x0 = nonzero_x[0]
    x1 = nonzero_x[-1]


    X=int(img_R.shape[1])
    Y=int(img_R.shape[0])

    cv2.line(img_R, (x0, y), (x0, y), (0, 255, 0), 20)
    cv2.line(img_R, (x1, y), (x1, y), (0, 255, 0), 20)

    midLane = (x1 + x0) / 2

    cv2.line(img_R, (midLane, y), (midLane, y), (0, 0, 255), 10) 
    cv2.line(img_R, (X/2,0),(X/2,Y),(255,0,0),5)

    # print( x0, midLane, x1 )
    
    cv2.imshow("asdf", img_R)

    cv.imshow("source",src)
    # cv.imshow("Detected Lines(in red)-Standard Hough Line Transform",cdst)
    cv.imshow("Detected Lines(in red)-Probabilistic Line Transform",cdstP)

    if cv2.waitKey(0)& 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
