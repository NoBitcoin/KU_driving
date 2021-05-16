# 0516 -> range define , for 7 -> 9
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

    left_max_left = []
    left_max_right = []
    right_max_left = []
    right_max_right = []

    k = 0
    for j in range(0,9):
        nonzero_left_sum = []
        nonzero_right_sum = []
        
        for i in range(x):
        # every x
            if( i >= 0 and i < middle_x - margin):
                nonzero_y = image[ 0 : y - k , i ].nonzero()[0]
                nonzero_left = len(nonzero_y)
                nonzero_left_sum.append(nonzero_left)
                # print(nonzero_left_sum)
            elif( i <= x and i > middle_x + margin):
                nonzero_y = image[ 0 : y - k , i ].nonzero()[0]
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

        # next rectangle range definition
        # case left
        if(j == 0):
            cv2.rectangle(image, (Left_Max - 20, y - k), (Left_Max + 20, y - 40 - k), (0,255,0), 1)
            left_max_left.append(Left_Max - 20)
            left_max_right.append(Left_Max + 20)
        else:
            if((Left_Max + 20 >= left_max_left[j - 1] and Left_Max + 20 <= left_max_right[j - 1]) or (Left_Max - 20 >= left_max_left[j - 1] and Left_Max - 20 <= left_max_right[j - 1])):
                cv2.rectangle(image, (Left_Max - 20, y - k), (Left_Max + 20, y - 40 - k), (0,255,0), 1)
                left_max_left.append(Left_Max - 20)
                left_max_right.append(Left_Max + 20)
            else:
                cv2.rectangle(image, (left_max_left[j - 1], y - k), (left_max_right[j - 1], y - 40 - k), (0,255,0), 1)
                left_max_left.append(left_max_left[j - 1])
                left_max_right.append(left_max_right[j - 1])
        # case right
        if(j == 0):
            cv2.rectangle(image, (Right_Max - 20, y - k), (Right_Max + 20, y - 40 - k), (0,255,0), 1)
            right_max_left.append(Right_Max - 20)
            right_max_right.append(Right_Max + 20)
        else:
            if((Right_Max + 20 >= right_max_left[j - 1] and Right_Max + 20 <= right_max_right[j - 1]) or (Right_Max - 20 >= right_max_left[j - 1] and Right_Max - 20 <= right_max_right[j - 1])):
                cv2.rectangle(image, (Right_Max - 20, y - k), (Right_Max + 20, y - 40 - k), (0,255,0), 1)
                right_max_left.append(Right_Max - 20)
                right_max_right.append(Right_Max + 20)
            else:
                cv2.rectangle(image, (right_max_left[j - 1], y - k), (right_max_right[j - 1], y - 40 - k), (0,255,0), 1)
                right_max_left.append(right_max_left[j - 1])
                right_max_right.append(right_max_right[j - 1])
        

        k += 40

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
    cv2.imshow('test', cdstP_bird)

    if cv2.waitKey(33)& 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()