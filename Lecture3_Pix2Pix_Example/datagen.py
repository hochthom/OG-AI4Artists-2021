# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:17:36 2021

@author: Thomas
"""


import os
import numpy as np
from PIL import Image



def load(image_file):
    img = Image.open(image_file).convert('RGB')
    return normalize(np.array(img, dtype=np.float32))

def normalize(image, centered=False):
    if centered:
        return  (image / 127.5) - 1
    return image / 255

def split(image):
    w = image.shape[1]
    w = w // 2
    target = image[:, w:, :]
    input_ = image[:, :w, :]
    return input_, target

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))



class DataGenerator():
    
    def __init__(self, images, image_dir, target_dir=None):
        'Initialization'
        self.images = [fn for fn in images if is_image_file(fn)]
        self.image_dir = image_dir
        self.target_dir = target_dir

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.images)

    def __getitem__(self, index):
        'Generate one batch of data'
        image = load(os.path.join(self.image_dir, self.images[index]))
        if self.target_dir is None:
            input_, target = split(image)
        else:
            input_ = image
            target = load(os.path.join(self.target_dir, self.images[index]))
        
        h, w = input_.shape[:2]
        input_ = input_.reshape((1, h, w, 3))
        target = target.reshape((1, h, w, 3))
        return input_, target

    def sample(self):
        return self.__getitem__(np.random.randint(len(self.images)))

    
            
if __name__ == '__main__':
    import tqdm

    TRAIN_IMAGE_DIR = 'data/train'
    imgs = os.listdir(TRAIN_IMAGE_DIR)
    train_dataset = DataGenerator(imgs, TRAIN_IMAGE_DIR)

    inp, tgt = train_dataset.sample()
    print(inp.shape)
    print(tgt.shape)
    