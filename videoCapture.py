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
    frame = cv2.flip(frame,0)
    frame = cv2.rotate(frame,0)
    frame = cv2.rotate(frame,0)
    
    frame = face.boxFrame(frame)
    frame = face.boxHand(frame)
    frame= face.boxHand2(frame)
    #frame = face.noise(frame)
    frame = cv2.resize(frame,None,fx=.5,fy=.5)
    
    """
    for i in range(0,frame.shape[0]):
        for k in range(0, frame.shape[1]):
            frame[i,k] = frame[i,k]+randint(0,200)
    """
    if face.handFaceIntersection(frame):
        print("INTERSECTION")
    
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
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()