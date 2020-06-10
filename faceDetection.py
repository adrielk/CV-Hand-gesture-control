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

#PLaces boxes around faces an displays it, used for still images only
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

#Helper function that draws boxes given coordinates (face_rects) and vertex colors (dotC) and rectangle color (rectC)
def drawBox(img_raw, face_rects, dotC, rectC, text):
    for(x,y,w,h)in face_rects:
        point1 = (x,y)
        point2 = (x,y+h)
        point3 = (x+w,y)
        point4 = (x+w,y+h)
        mid = (x,y-15)
        rad = 5
        color = dotC
        fontColor = (255,255,255)
        thickness = -1
        font = cv2.FONT_HERSHEY_COMPLEX
        fontScale = 0.5
        cv2.putText(img_raw,text, mid,font,fontScale,fontColor,1)
        cv2.putText(img_raw,str(point1),point1,font,fontScale,fontColor,1)
        cv2.putText(img_raw,str(point2),point2,font,fontScale,fontColor,1)
        cv2.putText(img_raw,str(point3),point3,font,fontScale,fontColor,1)
        cv2.putText(img_raw,str(point4),point4,font,fontScale,fontColor,1)
        cv2.rectangle(img_raw,(x,y),(x+w,y+h),rectC)
        cv2.circle(img_raw, point1,rad,color,thickness)
        cv2.circle(img_raw, point2,rad,color,thickness)
        cv2.circle(img_raw, point3,rad,color,thickness)
        cv2.circle(img_raw, point4,rad,color,thickness)
    return img_raw
   
#Places boxs around faces, used for frames of video capture
def boxFrame(img_raw):
    img_gray = convertToGrey(img_raw)
    img_rgb = img_raw
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1,minNeighbors = 5)
    dotC = (0,0,255)
    rectC = (0,255,0)
    drawBox(img_rgb, faces_rects, dotC, rectC,'Face')
    
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
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 4)
    dotC = (0,0,255)
    rectC = (0,255,0)
    drawBox(img_rgb, faces_rects, dotC, rectC,'Palm')
    
    return img_rgb

def boxHand2(img_raw):
    img_gray = convertToGrey(img_raw)
    img_rgb = img_raw
    haar_cascade_face = cv2.CascadeClassifier('aGest.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 10)
    dotC = (255,0,0)
    rectC = (0,255,255)
    drawBox(img_rgb, faces_rects, dotC, rectC)
        
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

#Detects presence of face without drawing box
def detectFace(img_raw):
    img_gray = convertToGrey(img_raw)
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 5)

    return (len(faces_rects)>0)#True means face, False means no face

#Checks if point (bounded) is contained within a rectangle (boundary1 and boundary2)
def bounded(bounded, boundary1, boundary2):
    boundedX = bounded[0]
    boundedY = bounded[1]
    boundary1X = boundary1[0]
    boundary2X = boundary2[0]
    boundary1Y = boundary1[1]
    boundary2Y = boundary2[1]
    
    if(boundedX<boundary2X and boundedX>boundary1X and boundedY>boundary1Y and boundedY<boundary2Y):
        return True
    
    return False
       
#Creates a rectangle which represents the area of intersection of two rectangles 
def fillIntersection(img, origin, offsets):
    originX = int(origin[0])
    originY = int(origin[1])
    offsetX = int(offsets[0])
    offsetY = int(offsets[1])
    
    cv2.rectangle(img,origin,(originX+offsetX,originY+offsetY),(0,0,255), -1)
    return img

def mirrorImage(img):
    frame = cv2.flip(img,0)
    frame = cv2.rotate(frame,0)
    frame = cv2.rotate(frame,0)
    return frame

def checkIntersectionAndFill(img_raw,faceBounds,fistBounds):
    #[(x,y),(x+w,y+h),(x,y+h),(x+w,y)]
    faceX = faceBounds[0][0]
    faceX2 = faceBounds[1][0]
    faceY = faceBounds[0][1]
    faceY2 = faceBounds[1][1]
    #rev = -1 if rev == True else 1
    intersects = False
    
    if bounded(fistBounds[0], faceBounds[0], faceBounds[1]):
        fB = fistBounds[0]
        xLength = abs(int(fB[0])-faceX2)
        yLength = abs(int(fB[1])-faceY2)
        offsets = (xLength, yLength)
        img_raw = fillIntersection(img_raw,fB, offsets)
        intersects = True
    elif bounded(fistBounds[1], faceBounds[0], faceBounds[1]):
        fB = fistBounds[1]
        xLength = -abs(int(fB[0])-faceX) 
        yLength = -abs(int(fB[1])-faceY)
        offsets = (xLength, yLength)
        img_raw = fillIntersection(img_raw,fB, offsets)
        intersects = True
    elif bounded(fistBounds[2], faceBounds[0], faceBounds[1]):
        fB = fistBounds[2]
        xLength = abs(int(fB[0])-faceX2)
        yLength = -abs(int(fB[1])-faceY)
        offsets = (xLength, yLength)
        img_raw = fillIntersection(img_raw,fB, offsets)
        intersects = True
    elif bounded(fistBounds[3], faceBounds[0], faceBounds[1]):
        fB = fistBounds[3]
        xLength = -abs(int(fB[0])-faceX)
        yLength = abs(int(fB[1])-faceY2)
        offsets = (xLength, yLength)
        img_raw = fillIntersection(img_raw,fB, offsets)
        intersects = True
    
    return intersects
"""
def checkIntersectionAndFillReverse(img_raw,faceBounds,revFaceBounds,fistBounds):
    faceX = faceBounds[0][0]
    faceX2 = faceBounds[1][0]
    faceY = faceBounds[0][1]
    faceY2 = faceBounds[1][1]
    
    faceXRev = revFaceBounds[0][0]
    faceX2Rev = revFaceBounds[1][0]
    faceYRev = revFaceBounds[0][1]
    faceY2Rev = revFaceBounds[1][1]
    
    if bounded(fistBounds[0], revFaceBounds[0], revFaceBounds[1]):
        fB = fistBounds[0]
        xLength = abs(int(fB[0])-faceX2)
        yLength = abs(int(fB[1])-faceY2)
        offsets = (xLength, yLength)
        img_raw = fillIntersection(img_raw,fB, offsets)
        intersects = True
    elif bounded(fistBounds[1], revFaceBounds[0], revFaceBounds[1]):
        fB = fistBounds[1]
        xLength = -abs(int(fB[0])-faceX) 
        yLength = -abs(int(fB[1])-faceY)
        offsets = (xLength, yLength)
        img_raw = fillIntersection(img_raw,fB, offsets)
        intersects = True
    elif bounded(fistBounds[2], revFaceBounds[0], revFaceBounds[1]):
        fB = fistBounds[2]
        xLength = abs(int(fB[0])-faceX2)
        yLength = -abs(int(fB[1])-faceY)
        offsets = (xLength, yLength)
        img_raw = fillIntersection(img_raw,fB, offsets)
        intersects = True
    elif bounded(fistBounds[3], revFaceBounds[0], revFaceBounds[1]):
        fB = fistBounds[3]
        xLength = -abs(int(fB[0])-faceX)
        yLength = abs(int(fB[1])-faceY2)
        offsets = (xLength, yLength)
        img_raw = fillIntersection(img_raw,fB, offsets)
        intersects = True
"""

#Checks for intersection between hand and face. Fills in intersection plane
def handFaceIntersection(img_raw):
    img_gray = convertToGrey(img_raw)
    #img_reversed = mirrorImage(img_gray)
        
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    #haar_cascade_palm = cv2.CascadeClassifier('palm.xml')
    haar_cascade_fist = cv2.CascadeClassifier('aGest.xml')
    
    face_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 5)
    #palm_rects = haar_cascade_palm.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 4)
    fist_rects = haar_cascade_fist.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 10)
    
    faceBounds = [(0,0),(0,0)]
    #palmBounds = [(0,0),(0,0),(0,0),(0,0)]
    fistBounds = [(0,0),(0,0),(0,0),(0,0)] #can just np.zeros...
    fistRevBounds = [(0,0),(0,0),(0,0),(0,0)]
    faceRevBounds = [(0,0),(0,0)]
    
    for(x,y,w,h) in face_rects:
        faceBounds = [(x,y),(x+w,y+h)]
        
    """   
    for(x,y,w,h) in palm_rects:
        palmBounds = [(x,y),(x+w,y+h),(x,y+h),(x+w,y)]
    """
    for(x,y,w,h) in fist_rects:
        fistBounds = [(x,y),(x+w,y+h),(x,y+h),(x+w,y)]
    
        
    intersect = checkIntersectionAndFill(img_raw,faceBounds,fistBounds)
    
    dotC = (0,0,255)
    rectC = (0,255,0)
    img_raw = drawBox(img_raw, face_rects, dotC, rectC,'Face')

    dotCFist = (255,0,0)
    rectCFist = (0,255,255)
    img_raw = drawBox(img_raw, fist_rects, dotCFist, rectCFist,'Fist')
    
    
    img_reversed = mirrorImage(img_gray)#mirrorImage(img_raw)
    fist_rects_rev = haar_cascade_fist.detectMultiScale(img_reversed, scaleFactor = 1.1, minNeighbors = 10)
    face_rects_rev = haar_cascade_face.detectMultiScale(img_reversed, scaleFactor = 1.1, minNeighbors = 5)
    
    for(x,y,w,h) in fist_rects_rev:
        fistRevBounds = [(x,y),(x+w,y+h),(x,y+h),(x+w,y)]
    for(x,y,w,h) in face_rects_rev:
        faceRevBounds = [(x,y),(x+w,y+h)]
    img_raw = mirrorImage(img_raw)    
    intersect2 = checkIntersectionAndFill(img_raw,faceRevBounds,fistRevBounds)
    img_raw = drawBox(img_raw, fist_rects_rev, dotCFist, rectCFist,'Fist')  
    img_raw = mirrorImage(img_raw) 
    
    return (intersect or intersect2), img_raw
    
'''
Image convolution 
'''

def applyFilter(img, kernel):
    return cv2.filter2D(img, -1, kernel)
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

#pollutes image with random noise
def noise(img_raw):
    noiseKernel = (np.ones((3,3),np.float32))+randint(0,100)
    img_noisy = cv2.filter2D(img_raw,-1, noiseKernel)
    return img_noisy

def emboss(img_raw):
    eKernel = np.array([-2,-1,0,-1,1,1,0,1,2])
    return applyFilter(img_raw, eKernel)

def bottomSobel(img_raw):
    kernel = np.array([-1,-2,-1,0,0,0,1,2,1])
    return applyFilter(img_raw, kernel)

def outline(img_raw):
    kernel = np.array([-1,-1,-1,-1,8,-1,-1,-1,-1])
    return applyFilter(img_raw, kernel)

def edgeDetect(img_raw):
    kernel1 = np.array([1,0,-1,2,0,-2,1,0,-1])
    kernel2 = np.array([1,2,1,0,0,0,-1,-2,-1])
    return applyFilter(applyFilter(img_raw,kernel1),kernel2)

def applyHue(img_raw):
    lower_red = np.array([10,50,50])
    upper_red = np.array([130,255,255])
    img_raw = cv2.cvtColor(img_raw,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_raw, lower_red, upper_red)
    res = cv2.bitwise_and(img_raw,img_raw, mask= mask)
    return img_raw


