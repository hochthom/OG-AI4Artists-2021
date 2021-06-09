# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:34:11 2021

@author: Thomas
"""

import cv2
import json
import pickle
import socket
import numpy as np
from stylegan2 import StyleGAN


#DATA_DIR = '/fashion-gen/'
DATA_DIR = 'D:/ART/AI-4-Artists/class_4_Gen-Models_Img'
LOAD_IMG_STYLES = False
HOST, PORT = "localhost", 65432


model = StyleGAN(lr=0.0001, silent=False, path=DATA_DIR)
model.load(9)

with open('styles.pkl', 'rb') as fp:
    img_style = pickle.load(fp)
    img_noise = pickle.load(fp)

N_styles = len(img_noise)
print(N_styles, 'styles loaded')
S1, S2 = np.random.permutation(range(N_styles))[:2]


def generate_image_style(style, n2, ix1, ix2, vec):
    x = np.clip(vec['x'] + 1, 0, 2) / 2
    y = np.clip(vec['y'] + 1, 0, 2) / 2
    n2_im = n2[ix1] * y + n2[ix2] * (1 - y)

    w_space = []
    for i in range(len(style)):
        s1 = style[i][ix1, :]
        s2 = style[i][ix2, :]
        w_space.append(np.array([s1 * x + s2 * (1 - x)]))

    return w_space, np.array([n2_im])

# create initial image (for testing and initialisation)
vec = {'x':0, 'y':0}
w_space, n2im = generate_image_style(img_style, img_noise, S1, S2, vec)
imgs = model.generate_images(w_space, n2im)


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            mesg = conn.recv(1024)
            if len(mesg) > 0:
                mesg = mesg[:mesg.find(b"}") + 1]
                print(mesg)
                vec = json.loads(mesg)
                # create style
                w_space, n2im = generate_image_style(img_style, img_noise, S1, S2, vec)
                img = model.generate_images(w_space, n2im)
                img = np.uint8(np.clip(img[0], 0.0, 1.0) * 255)
                im_bgr = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)

                cv2.imshow('StyleGAN sample', im_bgr)

            if cv2.waitKey(5) & 0xFF == 27:
                break
