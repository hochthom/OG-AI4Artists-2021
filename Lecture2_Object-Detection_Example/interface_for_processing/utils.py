# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:23:10 2021

@author: Thomas
"""


kp_idx = {'RIGHT_EYE':0, 'LEFT_EYE':1, 'NOSE_TIP':2, 'MOUTH_CENTER':3,
          'RIGHT_EAR_TRAGION':4, 'LEFT_EAR_TRAGION':5}

def get_key_point(detection, kp_type):
    kp = detection.location_data.relative_keypoints[kp_idx[kp_type]]
    return {'id':kp_type, 'x':kp.x, 'y':kp.y}

def get_all_key_points(detection):
    return [get_key_point(detection, kp) for kp in kp_idx.keys()]

def get_bounding_box(detection):
    bb = detection.location_data.relative_bounding_box
    return {'xmin':bb.xmin, 'ymin':bb.ymin, 'width':bb.width, 'height':bb.height}

def calc_displacement(detection):
    nose = get_key_point(detection, 'NOSE_TIP')
    bbox = get_bounding_box(detection)
    xc = bbox['xmin'] + bbox['width'] / 2
    yc = bbox['ymin'] + bbox['height'] / 2
    dx = 2 * (nose['x'] - xc) / bbox['width']
    dy = 2 * (nose['y'] - yc) / bbox['height']
    return dx, dy
