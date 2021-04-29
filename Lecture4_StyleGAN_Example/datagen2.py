# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:14:16 2020

@author: Thomas
"""

import os
import zipfile
from PIL import Image
import numpy as np
import random



def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))


class DataGenerator(object):

    def __init__(self, image_source, data_dir, im_size, flip=True, verbose=True):
        self.data_dir = data_dir
        self.im_size = im_size
        self.flip = flip
        self.verbose = verbose
        self.images = []

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        
        data_file = os.path.join(self.data_dir, "imgs-%i.npy" % (self.im_size))
        if not os.path.exists(data_file):
            if os.path.isdir(image_source):
                self.folder_to_npy(image_source)
            elif image_source.endswith('.zip'):
                self.zip_to_npy(image_source)
            else:
                print('Warning: Could not create data file for training!')
            
        self.load_from_npy()


    def folder_to_npy(self, image_dir):
        if self.verbose:
            print("Converting from images to numpy file ...")

        names = []
        for dirpath, dirnames, filenames in os.walk(image_dir):
            for filename in [f for f in filenames if is_image_file(f)]:
                fname = os.path.join(dirpath, filename)
                names.append(fname)

        np.random.shuffle(names)
        if self.verbose:
            print(str(len(names)) + " images.")

        images = []
        for fname in names:
            try:
                temp = self.preprocess(Image.open(fname))
                images.append(np.array(temp, dtype='uint8'))
            except:
                print("Importing image failed on", fname)

        np.save(os.path.join(self.data_dir, "imgs-%i.npy" % (self.im_size)), np.array(images))

        
    def zip_to_npy(self, image_source):
        if self.verbose:
            print("Extracting images from zip file ...")

        with zipfile.ZipFile(image_source) as zfp:

            images = []
            for name in zfp.namelist():
                if is_image_file(name):
                    img_data = zfp.open(name)
                    try:
                        temp = self.preprocess(Image.open(img_data))
                        images.append(np.array(temp, dtype='uint8'))
                    except:
                        print("Importing image failed on", name)

            np.save(os.path.join(self.data_dir, "imgs-%i.npy" % (self.im_size)), np.array(images))


    def preprocess(self, image):
        return image.convert('RGB').resize((self.im_size, self.im_size), Image.BILINEAR)


    def load_from_npy(self):
        if self.verbose:
            print("Loading images to numpy file ...")
            
        data_file = os.path.join(self.data_dir, "imgs-%i.npy" % (self.im_size))
        if os.path.exists(data_file):
            self.images = np.load(data_file)
        else:
            print('Warning: Could not load training data from %s!' % data_file)

        if self.verbose:
            print("%i images loaded!" % len(self.images))

    def get_batch(self, num):
        idx = np.random.randint(0, self.images.shape[0] - 1, num)
        out = []
        for i in idx:
            out.append(self.images[i])
            if self.flip and random.random() < 0.5:
                out[-1] = np.flip(out[-1], 1)

        return np.array(out).astype('float32') / 255.0


