# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:29:57 2021

@author: Thomas
"""

import os
import cv2
import numpy as np


IMG_DIR = 'seq'

imgs = sorted(os.listdir(IMG_DIR))
 
out = None
for filename in imgs:
    img = cv2.imread(os.path.join(IMG_DIR, filename))
    height, width, layers = img.shape
    
    if out is None:
        out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 
                              15, (width, height))
    out.write(img)
out.release()

