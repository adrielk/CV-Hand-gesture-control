# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:17:41 2020

@author: thead
"""
from random import randint
import faceDetection as face
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
size = 3
intensity = 1

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = face.mirrorImage(frame)
    intersects, new_img = face.handFaceIntersection(frame)
    frame = new_img
    
    
    """
    if(face.detectHandBox(frame) and face.detectFace(frame)):
        intensity +=0.1
        #frame = face.intensify(frame,intensity)
    elif(face.detectHandBox2(frame) and face.detectFace(frame)):
        size+=2
        frame = frame * np.full_like(frame,20)
        #frame = face.smoothing(frame,size)
    else:
        size = 3
        intensity = 1
    """
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()