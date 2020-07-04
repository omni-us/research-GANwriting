import cv2
import numpy as np
import doc_augment_lib as da
import random
import glob

cv2.namedWindow('Display',0)
imgs = glob.glob('./imgs/*.png')
for i in imgs:
    img = cv2.imread(i, 0)
    img = np.float32(img)/ 255
    h, w = img.shape
    rotate_factor = h / w
    if h > w:
        thin_flag = True
    else:
        thin_flag = False
    for j in range(10):
        r=random.choice([0,1,2])
        if r==0:
            blur = da.LensBlur(img,lens_blur=(0.0,2.0))
        elif r==1:
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
            cv2.imshow('Display',np.vstack((img,blur,elastic,sheared,rotated,gamma,back)))
        else:
            cv2.imshow('Display', np.vstack((img, blur, elastic)))
        k=cv2.waitKey()
        if chr(k&255) == 'q':
            break
