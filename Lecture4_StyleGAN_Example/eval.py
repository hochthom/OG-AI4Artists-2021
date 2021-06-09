
import os
from PIL import Image
import numpy as np
import pickle


from stylegan2 import StyleGAN


DATA_DIR = '.'
TAG = 'seq'
#DATA_DIR = 'D:/ART/AI-4-Artists/styleGAN2/results/metFaces/'
LOAD_IMG_STYLES = True


model = StyleGAN(lr=0.0001, silent=False)#, path=DATA_DIR)
model.load(9)


if LOAD_IMG_STYLES and os.path.exists(os.path.join(DATA_DIR, 'styles.pkl')):
    with open(os.path.join(DATA_DIR, 'styles.pkl'), 'rb') as fp:
        img_style = pickle.load(fp)
        img_noise = pickle.load(fp)
else:
    img_style, img_noise = model.create_image_styles(100)
    with open(os.path.join(DATA_DIR, 'styles.pkl'), 'wb') as fp:
        pickle.dump(img_style, fp, -1)
        pickle.dump(img_noise, fp, -1)

if TAG.lower().startswith('seq'):
    imgs = model.generate_image_seq(img_style, img_noise, 19, 46, n_imgs=200)
else:
    imgs = model.generate_images(img_style, img_noise)

print("Saving images ...")
if not os.path.exists(os.path.join(DATA_DIR, TAG)):
    os.mkdir(os.path.join(DATA_DIR, TAG))
    
for i in range(len(imgs)):
    c1 = np.clip(imgs[i], 0.0, 1.0)
    x = Image.fromarray(np.uint8(c1*255))
    x.save(os.path.join(DATA_DIR, TAG, "image_%05i.png" % i))

