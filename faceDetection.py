# -*- coding: utf-8 -*-
"""
Haar Cascade xml file source:
https://github.com/parulnith/Face-Detection-in-Python-using-OpenCV/tree/master/data/haarcascades
https://github.com/Aravindlivewire/Opencv/blob/master/haarcascade/aGest.xml
@author: Adriel Kim
"""
from random import randint
import numpy as np
import cv2
import matplotlib.pyplot as plt

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convertToGrey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#PLaces boxes around faces, used for still images only
def boxFaces(img_name):
    img_raw = cv2.imread(img_name)
    img_gray = convertToGrey(img_raw)
    img_rgb = convertToRGB(img_raw)
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1,minNeighbors = 5)
    print("Faces found: ", len(faces_rects))
    print(faces_rects)
    
    for(x,y,w,h)in faces_rects:
        cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,255,0),4)
    plt.imshow(img_rgb)
   
#Places boxs around faces, used for frames of video capture
def boxFrame(img_raw):
    img_gray = convertToGrey(img_raw)
    img_rgb = img_raw
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1,minNeighbors = 5)
    #print("Faces found: ", len(faces_rects))
   # print(faces_rects)
    
    for(x,y,w,h)in faces_rects:
        cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,255,0),2)
    
    return img_rgb

def boxSmile(img_raw):
    img_gray = convertToGrey(img_raw)
    img_rgb = img_raw
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_smile.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray,scaleFactor = 1.7, minNeighbors = 25)

    for(x,y,w,h)in faces_rects:
        cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,0,255),2)
    
    return img_rgb

def boxEyes(img_raw):
    img_gray = convertToGrey(img_raw)
    img_rgb = img_raw
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_eye.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.3, minNeighbors = 6)

    for(x,y,w,h)in faces_rects:
        cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(255,0,0),2)
    
    return img_rgb

def boxBody(img_raw):
    img_gray = convertToGrey(img_raw)
    img_rgb = img_raw
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray)#, scaleFactor = 1.5, minNeighbors = 6)

    for(x,y,w,h)in faces_rects:
        cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(255,255,0),2)
    
    return img_rgb

def boxHand(img_raw):
    img_gray = convertToGrey(img_raw)
    img_rgb = img_raw
    haar_cascade_face = cv2.CascadeClassifier('palm.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 5)

    for(x,y,w,h)in faces_rects:
        cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,0,255),2)
    
    return img_rgb

def boxHand2(img_raw):
    img_gray = convertToGrey(img_raw)
    img_rgb = img_raw
    haar_cascade_face = cv2.CascadeClassifier('aGest.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 10)

    for(x,y,w,h)in faces_rects:
        cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,255,255),2)
    
    return img_rgb
#Frontal facing fist detection
def detectHandBox(img_raw):
    img_gray = convertToGrey(img_raw)
    haar_cascade_face = cv2.CascadeClassifier('aGest.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 10)
    
    return (len(faces_rects)>0)#True means hand, False means no hand
#Palm detection
def detectHandBox2(img_raw):
    img_gray = convertToGrey(img_raw)
    haar_cascade_face = cv2.CascadeClassifier('palm.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 10)

    return (len(faces_rects)>0)#True means hand, False means no hand

def detectFace(img_raw):
    img_gray = convertToGrey(img_raw)
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 5)

    return (len(faces_rects)>0)#True means face, False means no face

def handFaceIntersection(img_raw):
    img_gray = convertToGrey(img_raw)
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    haar_cascade_palm = cv2.CascadeClassifier('palm.xml')
    haar_cascade_fist = cv2.CascadeClassifier('aGest.xml')
    
    face_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 5)
    palm_rects = haar_cascade_palm.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 10)
    fist_rects = haar_cascade_fist.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 10)
    
    faceBounds = [(0,0),(0,0)]
    palmBounds = [(0,0),(0,0),(0,0),(0,0)]
    fistBounds = [(0,0),(0,0),(0,0),(0,0)]
    
    intersect = False    
    for(x,y,w,h) in face_rects:
        faceBounds = [(x,y),(x+w,y+h)]
        
    for(x,y,w,h) in palm_rects:
        palmBounds = [(x,y),(x+w,y+h),(x,y+h),(x+w,y)]
        
    for(x,y,w,h) in fist_rects:
        fistBounds = [(x,y),(x+w,y+h),(x,y+h),(x+w,y)]
        
        
        
    faceX = faceBounds[0][0]
    faceX2 = faceBounds[1][0]
    faceY = faceBounds[1][0]
    faceY2 = faceBounds[1][1]
    
    #check all for points and see if any lie within the face's box
    for i in range(0,len(fistBounds)):
        fistX = fistBounds[i][0]
        fistY = fistBounds[i][1]
        #print("Fist: ",fistX,",",fistY)
        #print("Face",faceX, ",",faceY)
        if(fistX<faceX2 and fistX>faceX and fistY<faceY and fistY>faceY2):
            intersect = True
            break
        
    return intersect
    
    
#Smoothing convolution filter, applies average blur
def smoothing(img_raw,size):
    kernel = (np.ones((size,size),np.float32)/(size*size)) #averaging kernel
    img_smooth = cv2.filter2D(img_raw,-1,kernel)
    return img_smooth

#Intensifying convolution filter. Brightens image
def intensify(img_raw,intensity):
    kernel = (np.ones((5,5),np.float32)/25)*intensity #averaging kernel
    img_smooth = cv2.filter2D(img_raw,-1,kernel)
    return img_smooth

#Sharpens image
def sharpen(img_raw):
    sharpenKernel = [0,-1,0,-1,5,-1,0,-1,0]
    npKernel = np.array(sharpenKernel)
    img_smooth = cv2.filter2D(img_raw,-1,npKernel)
    return img_smooth

def noise(img_raw):
    noiseKernel = (np.ones((3,3),np.float32))+randint(0,100)
    img_noisy = cv2.filter2D(img_raw,-1, noiseKernel)
    return img_noisy
"""
plt.imshow(img_rgb)
font = cv2.FONT_HERSHEY_SIMPLEX
text = cv2.putText(img_rgb,'hello',(100,300), font, 1,(255,255,255),2,cv2.LINE_AA)
plt.imshow(text)
"""