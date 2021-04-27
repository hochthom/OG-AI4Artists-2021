# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:10:04 2021

@author: Thomas
"""

import cv2
import json
import mediapipe as mp
import socket

from utils import get_landmarks, calc_arm_movement


HOST = '127.0.0.1'  # The server's IP address
PORT = 65432        # The port used by the server


# Prepare detection alg for faces
pose_detector = mp.solutions.pose.Pose(min_detection_confidence=0.5)

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
#drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    
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
        results = pose_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Draw the face detection annotations on the image.
        if results.pose_landmarks is not None:
            kpts = get_landmarks(results.pose_landmarks)
            lft, rgt = calc_arm_movement(kpts)
            mesg = json.dumps({'x':lft, 'y':rgt})
            s.sendall(mesg.encode())
        
        # Draw the pose annotations on the image.
        mp_drawing.draw_landmarks(image, results.pose_landmarks, 
                                  mp.solutions.pose.POSE_CONNECTIONS)
    
        cv2.imshow('MediaPipe Pose Recognition', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
