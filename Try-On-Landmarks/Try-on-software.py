# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 20:24:47 2021

@author: samiha hussein
"""

import numpy as np
import cv2
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
glasses_image = cv2.imread("images/glass2-png.png")

# Detect face
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

array_landmarks= [0,17,18,19,20,21,27,22,23,24,25,26,16]
    

while True:
    _,frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    
    faces = detector(frame)
    
    for face in faces:
        landmarks = predictor(gray_frame, face)
        
        for index in array_landmarks:
            left_face = (landmarks.part(index).x ,landmarks.part(index).y)
            
            cv2.circle(frame, (left_face),3, (255,0,0),-1)
        
            first = (landmarks.part(0).x, landmarks.part(0).y)
            last = (landmarks.part(16).x, landmarks.part(16).y)
            center = (landmarks.part(27).x, landmarks.part(27).y)
            
            #convert to int(hypot) if it's possible
            face_width = int(hypot(first[0] - last[0],
                                   first[1] - last[1])  )
            face_height = int(face_width * 0.403)
            print(face_width)
            print("height")
            print(face_height)
            
            top_left = (int(center[0] - face_width / 2),
                                  int(center[1] - face_height / 2))
            top_right =  (int(center[0] + face_width/2), 
                                   int(center[1] + face_height/2))
            
            # rectangle to check if the area is right (where we will put the glasses)
            #cv2.rectangle(frame, top_left,top_right,(0,255,0), 2)
            
                    
            glasses_on = cv2.resize(glasses_image,(face_width, face_height) )
            glasses_on_gray = cv2.cvtColor(glasses_on, cv2.COLOR_BGR2GRAY)
            _, glasses_mask = cv2.threshold(glasses_on_gray,25,255,cv2.THRESH_BINARY_INV)
            
            
            # area where we will put the glasses on
            eye_area = frame[top_left[1]:top_left[1] + face_height,
                                 top_left[0]:top_left[0] + face_width ]
            
            eye_no_area = cv2.bitwise_and(eye_area, eye_area, 
                                          mask= glasses_mask)
            
            final_glasses = cv2.add(eye_no_area,glasses_on)
            
            
            
            frame[top_left[1]:top_left[1] + face_height,
                                 top_left[0]:top_left[0] + face_width ] = final_glasses
        
        
        # it grows everytime i zoom in or out from the camera
        #cv2.imshow("glasses:", glasses_on)
        #cv2.imshow("GLASSES ON", glasses_mask)
        #cv2.imshow("final eye:",final_glasses)
        #cv2.imshow("Eye Area", eye_area)
        
        
    cv2.imshow("Try-On", frame)
    #cv2.imshow("glasses", glasses_image)
    
    
    key = cv2.waitKey(5)
    
    if key ==27:
        break;
        
#3shan el frame may3ala2sh
cv2.destroyAllWindows();