#! /usr/bin/env python3

# https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html
# https://github.com/abidrahmank/OpenCV2-Python-Tutorials


import cv2
import numpy as np
import sys


def read_grayscale(img_path):
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0


def save_image(img_path, img):
    cv2.imwrite(img_path, img * 255)


def get_image_depth(img):
    if img.dtype == np.float64:
        return cv2.CV_64F
    return None


def clip(img):
    return np.clip(img, 0, 1)


if __name__ == "__main__":
    print("Reading the image")
    image = read_grayscale("building.tiff")

    print("Saving grayscale image")
    save_image("building_grayscale.jpg", image)

    image_depth = get_image_depth(image)
    print("Current image depth is %s" % image.dtype)

    print("Applying Gaussian blur")
    image_blurred = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=1)

    print("Applying Laplacian filter")
    image_laplacian = cv2.Laplacian(image_blurred, image_depth)

    print("Adding some contrast to the image with highpass filter")
    save_image("building_grayscale_laplacian_1.jpg", clip(image_laplacian * 7 + 0.5))

    height, width = image_laplacian.shape
    for i in range(0,height):
        for j in range(0,width):
            sys.stdout.write("%s " % image_laplacian[i][j])
        print ""

    # Image is too dark so add some contrast
    print("Adding some contrast to the image with highpass filter")
    image_laplacian = cv2.filter2D(image_laplacian, ddepth=image_depth, kernel=np.array([
        [-1, -1, -1],
        [-1, 25, -1],
        [-1, -1, -1]
    ]))
    save_image("building_grayscale_laplacian_2.jpg", image_laplacian)
