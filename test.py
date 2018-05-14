#! /usr/bin/env python3

# https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html
# https://github.com/abidrahmank/OpenCV2-Python-Tutorials

import cv2
import numpy as np
import matplotlib.pyplot as plt

print(cv2.__version__)

#flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#print(flags)

img_bgr = cv2.imread('messi5.jpg')
print(type(img_bgr), img_bgr.dtype, img_bgr.shape)
cv2.imshow('BGR', img_bgr)
cv2.waitKey(0)

img_g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print(type(img_g), img_bgr.dtype, img_g.shape)
cv2.imshow('grey', img_g)
cv2.waitKey(0)

z = img_bgr
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        if 1.2*z[i,j,0]>z[i,j,1] or 1.2*z[i,j,2]>z[i,j,1]:
            z[i,j] = [img_g[i,j]]*3

cv2.imwrite('grass.jpg', z)
cv2.imshow('grass', z)
cv2.waitKey(0)
           
h = [0]*256
for i in img_g.flatten():
    h[i] += 1

plt.plot(h)
plt.savefig('histogram.png')
plt.show()

