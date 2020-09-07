# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 15:50:45 2020

@author: sefa
"""

import cv2
import numpy as np

haarcascade = "haarcascade_frontalface_alt2.xml"
haarcascade_clf = "data/" + haarcascade
detector = cv2.CascadeClassifier(haarcascade_clf)

LBFmodel = "LFBmodel.yaml"
LBFmodel_file = "data/" + LBFmodel
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel_file)

webcam_cap = cv2.VideoCapture(0)
while(True):
    _, frame = webcam_cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray)
    
    for (x,y,w,d) in faces:
        
        _, landmarks = landmark_detector.fit(gray, np.array(faces))
        
        for landmark in landmarks:
           
            for x,y in landmark[0]:
    
                cv2.circle(frame, (x, y), 1, (255, 0, 0), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(20) & 0xFF  == ord('q'):
        webcam_cap.release()
        cv2.destroyAllWindows()
        break
    


import turtle

n = 60
pen=turtle.Turtle()

pen.circle(10)
