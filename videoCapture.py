# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:17:41 2020

@author: Adriel Kim
"""
from random import randint
import faceDetection as face
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
size = 3
intensity = 1

effects = ['sharpen','noise', 'emboss', 'outline', 'edgeDetect']

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = face.mirrorImage(frame)
    intersects, new_img = face.handFaceIntersection(frame)
    frame = new_img
    randEffect = effects[randint(0,len(effects)-1)]
    
    
    if(face.detectHandBox(frame) == True and intersects == False):
        if(randEffect == 'sharpen'):
            frame = face.sharpen(frame)
        elif(randEffect == 'noise'):
            frame = face.noise(frame)
        elif(randEffect == 'emboss'):
            frame = face.emboss(frame)
        elif(randEffect == 'outline'):
            frame = face.outline(frame)
        else:
            frame = face.edgeDetect(frame)
            
        frame = face.intensify(frame,intensity)
        size += 3
        intensity += 1
    else:
        size = 3
        intensity = 1
    
    if(intersects == True):
        frame = face.applyHue(frame)
    
    

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()