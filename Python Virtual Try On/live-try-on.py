import cv2
import numpy as np
import dlib
from math import hypot

glasses_image = cv2.imread('images/glass2.png',-1)
image_mask = glasses_image[:,:,3]
image_inv_mask = cv2.bitwise_not(image_mask)
glasses_image = glasses_image[:,:,0:3]
originalGlasses_height, originalGlasses_width = glasses_image.shape[:2] 
video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("C:/xampp/htdocs/Python Virtual Try On/cascades/haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("C:/xampp/htdocs/Python Virtual Try On/cascades/haarcascade_eye.xml")


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    

    

    
    #x pos, y pos, width, height
    for (x, y, w, h) in faces:
        
        #Draws a blue rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        #roi: region of interest 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eyesCascade.detectMultiScale(roi_gray) 
        for (eyes_x,eyes_y,eyes_width,eyes_height) in eyes:
            
            #cv2.rectangle(roi_color, (eyes_x, eyes_y), (eyes_x+eyes_width, eyes_y+eyes_height), (0, 255, 0), 1)
            glassesWidth = 3 * eyes_width
            glassesHeight = (glassesWidth*(originalGlasses_height/originalGlasses_width))
            x1 = eyes_width - int(glassesWidth/4)
            x2 = x + eyes_width + int(glassesWidth/4)
            y1 = int(y/3) + eyes_height - int(glassesHeight/2)
            y2 = int(y/3) + eyes_height + int(glassesHeight/2)
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            glassesWidth = x2 - x1
            glassesHeight = y2 - y1
            glasses = cv2.resize(glasses_image, (glassesWidth,glassesHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(image_mask, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(image_inv_mask, (glassesWidth,glassesHeight),interpolation=cv2.INTER_AREA)
            roi = roi_color[y1:y2, x1:x2]
            roi_bg = cv2.bitwise_and(roi, roi,mask= mask_inv)
            roi_fg = cv2.bitwise_and(glasses,glasses, mask = mask)
            dst = cv2.add(roi_bg,roi_fg)
            roi_color[y1:y2, x1:x2] = dst
            break


    # Display the resulting frame
    cv2.imshow('Virtual Try On', frame)

    #Video stop key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()