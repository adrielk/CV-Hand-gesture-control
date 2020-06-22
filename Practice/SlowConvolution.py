# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 23:22:35 2020

@author: thead
"""

import cv2
import numpy as np


def showImg(img_name):
    img_raw = cv2.imread(img_name)
    cv2.imshow(img_name, img_raw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def SlowConvolution(img_name, kernel):
    img_raw = cv2.imread(img_name)
    img_temp = cv2.imread(img_name)
    imgX = img_raw.shape[0]
    imgY = img_raw.shape[1]
    imgZ = img_raw.shape[2]
    
    kernelDim = len(kernel)
    
    offset = int((1/2) * kernelDim - (1/2))#NOTE: Kernel dimensions are only odd-numbered!
    print(offset)
    #perform convolution by doing element wise mutliplication, and summing up the elements within kernel. Set current pixel to sum value.
    for i in range(offset, imgX-offset):
        for k in range(offset, imgY-offset):
            startIndR = i-offset
            startIndC = k-offset
           # print(startIndR,",", startIndC)
            for l in range(0, 3):
              
                sum = 0;
                for x in range(0, kernelDim):
                    for y in range(0, kernelDim):
                        kernelIndX = x#- startIndR
                        kernelIndY = y #- startIndC
                        sum += kernel[kernelIndX][kernelIndY] * img_raw[startIndR+x][startIndC+y][l]
                        #print(x+startIndR,",", y+startIndC)
                        
                img_temp[i][k][l] = sum
                #img_raw[i][k][l] = sum
                #img_raw[k][i][l] = img_raw[i][k][l]#sum
                    
    
    scale_percent = 200 # percent of original size
    width = int(img_temp.shape[1] * scale_percent / 100)
    height = int(img_temp.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img_temp, dim, interpolation = cv2.INTER_AREA)
    
    cv2.imshow(img_name, resized)
    cv2.waitKey(0)#this essentially adds a wait so the system doesn't crash...
    cv2.destroyAllWindows()
    
    
#Optimzed convolution algorithm with vectorizaton
def vectorConvolution(img_name, kernel):
    #Kernel vars
    vKernel = np.array(kernel)
    kernelDim = vKernel.shape[0]
    #img vars
    img_raw = cv2.imread(img_name)
    img_new = cv2.imread(img_name)
    imgX = img_raw.shape[0]
    imgY = img_raw.shape[1]
    imgZ = img_raw.shape[2]
        
    offset = int((1/2)*kernelDim - (1/2))
    
    for i in range(offset, imgX-offset):
        for k in range(offset, imgY - offset):
            for l in range(0, imgZ):
                startIndR = i - offset
                startIndC = k - offset
                #Current grid kernel is over
                cur_grid = np.array(img_raw[startIndR:startIndR+kernelDim,startIndC:startIndC+kernelDim,l]).reshape(kernelDim, kernelDim)
                # print("Current grid:\n",cur_grid, "\n")
                # print("Kernel:\n",vKernel,'\n')
                sum = np.sum(np.multiply(vKernel, cur_grid))
                img_new.itemset((i,k,l),sum)
                #print(img_new[i,k,l],"==",sum)
                # print(sum)
                # print("starting coords: (%d,%d)"% (startIndR, startIndC))
                # print("Starting value: ", img_raw[startIndR, startIndC,l])
                # print("center: (%d, %d)"%(i, k))
                # print("center value: ", img_raw[i,k,l])
                #img_new[i,k,l] = sum
                #img_new.append(sum)
                #print("Upated center: ", img_new[i,k,l], "==", sum)
                # print("Center before: ", img_raw[i,k,l], ", Center after: ", img_new[i,k,l])
                # print("____________________")

    #img_final = np.array(img_new).reshape(imgX-(offset*2),imgY-(offset*2),imgZ).astype(np.uint8)
    # scale_percent = 150 # percent of original size
    # width = int(img_final.shape[1] * scale_percent / 100)
    # height = int(img_final.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # # resize image
    # resized = cv2.resize(img_final, dim, interpolation = cv2.INTER_AREA)
    
    cv2.imshow(img_name,img_new)#resized)
    cv2.waitKey(0)#this essentially adds a wait so the system doesn't crash...
    cv2.destroyAllWindows()  

    
k = [[0,-1,0],[-1,5,-1],[0,-1,0]]
k2 = [[0,0,0],[0,5,0],[0,0,0]]
k3 = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
k4 = [[-2,-1,0],[-1,1,1],[0,1,2]]
#SlowConvolution('lena_color.png',k4)
vectorConvolution('lena_color.png',k3)




