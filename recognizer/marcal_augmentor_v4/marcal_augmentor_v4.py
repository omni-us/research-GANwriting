import cv2
import numpy as np
import marcal_augmentor_v4.doc_augment_lib as da
import random
import glob

# 0-1.
def augmentor(img):
    #img = np.float32(img)/ 255
    h, w = img.shape
    if w == 0 or h == 0:
        print("h, w: ", h, w)
        return img
    rotate_factor = h / w
    if h > w:
        thin_flag = True
    else:
        thin_flag = False
    r=random.choice([0,1,2])
    if r == 0:
        blur = da.LensBlur(img,lens_blur=(0.0,2.0))
    elif r == 1:
        blur = da.Sharpen(img,lens_blur=(0.0,2.0))
    else:
        blur = img
    #blur = da.LensBlur(img,lens_blur=(0.0,2.0))
    elastic = da.ElasticTransform(blur,alpha=1750, sigma=45)
    if not thin_flag:
        sheared = da.ShearNoPad(elastic,shear=(-.5,.25))
        rotated = da.RotationNoPad(sheared,rotation=(-5.0*rotate_factor,5.0*rotate_factor))
        gamma = da.GammaCorrection(rotated,gamma=(.3,3.0))
        back = da.RandomBackground(gamma)
        return back
    else:
        return elastic

if __name__ == '__main__':
    cv2.namedWindow('Display',0)
    imgs = glob.glob('./imgs/*.png')
    for i in imgs:
        img = cv2.imread(i, 0)
        out = augmentor(img)
        cv2.imshow('Display',np.vstack((img, out)))
        k=cv2.waitKey()
        if chr(k&255) == 'q':
            break
