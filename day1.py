#! /usr/bin/env python3

# https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html
# https://github.com/abidrahmank/OpenCV2-Python-Tutorials


import cv2
import numpy as np
from scipy import ndimage


def random_gauss(image, sigma):
    return np.random.normal(0, sigma, image.shape)


def clip(image):
    return np.clip(image, 0, 1)


def random_saltpepper(image, sigma):
    return np.random.choice([-1, 0, 1], image.shape, p=[sigma / 2, 1 - sigma, sigma / 2])


def lowpass(image, n):
    mask = np.full((n, n), 0.5)
    height, width = image.shape
    for x in range(0, height):
        for y in range(0, width):
            


def highpass(image, n):
    mask = np.reshape([-1 for x in range(0, n * n)], (n, n))
    mask[int(n / 2)][int(n / 2)] = 100
    return ndimage.filters.convolve(image, mask)


if __name__ == "__main__":
    messi_tiff_img = cv2.imread("messi-gray.tiff", cv2.IMREAD_GRAYSCALE) / 255.0
    messi_tiff_img_gauss = clip(messi_tiff_img + random_gauss(messi_tiff_img, 0.05))
    cv2.imwrite("messi-test-gauss.jpg", messi_tiff_img_gauss * 255)
    messi_tiff_img_salt = clip(messi_tiff_img + random_saltpepper(messi_tiff_img, 0.05))
    cv2.imwrite("messi-test-salt.jpg", messi_tiff_img_salt * 255)
    messi_tiff_img_lowpass = clip(messi_tiff_img + lowpass(messi_tiff_img, 3))
    # print(lowpass(messi_tiff_img, 5))
    cv2.imwrite("messi-test-lowpass.jpg", messi_tiff_img_lowpass * 255)
    messi_tiff_img_higpass = clip(messi_tiff_img + highpass(messi_tiff_img, 3))
    # print(lowpass(messi_tiff_img, 5))
    cv2.imwrite("messi-test-highpass.jpg", messi_tiff_img_higpass * 255)
