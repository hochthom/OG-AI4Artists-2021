# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:10:04 2021

@author: Thomas
"""

import cv2
import mediapipe as mp


# Prepare detection alg for faces
face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
#drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# For webcam input:
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
   
    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # search for faces in the image
    results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Draw the face detection annotations on the image.
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
            break
    
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
