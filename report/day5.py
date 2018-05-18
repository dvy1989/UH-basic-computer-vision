#! /usr/bin/env python3

# https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html
# https://github.com/abidrahmank/OpenCV2-Python-Tutorials


from os import path

import cv2
import numpy as np

NUMBER_OF_READ_PAIRS = 0
FRAMES_FOLDER = "frames"


def read_image():
    global NUMBER_OF_READ_PAIRS
    img_path = path.join(FRAMES_FOLDER, "%s.jpeg" % str(NUMBER_OF_READ_PAIRS).zfill(8))
    print img_path
    NUMBER_OF_READ_PAIRS += 1
    if path.exists(img_path):
        return cv2.imread(img_path)
    else:
        return None


def read_pair_of_images():
    return read_image(), read_image()


if __name__ == "__main__":
    stereo = cv2.StereoSGBM_create(
        numDisparities=32,
        blockSize=7,
        P1=200,
        P2=500,
        disp12MaxDiff=20,
        preFilterCap=20,
        uniquenessRatio=1,
        speckleWindowSize=0,
        speckleRange=7,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    running = True
    while running:
        img_left, img_right = read_pair_of_images()
        # print NUMBER_OF_READ_PAIRS / 2
        if img_left is None or img_right is None:
            running = False
        else:
            # if NUMBER_OF_READ_PAIRS / 2 == 78:
            img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            disparity = stereo.compute(img_left_gray, img_right_gray)
            colored_disparity = cv2.cvtColor(np.array(disparity, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
            colored_disparity[disparity < 0] = (0, 0, 255)
            print img_left_gray.shape, colored_disparity.shape
            cv2.imwrite(path.join(FRAMES_FOLDER, "disparity%s.jpeg" % (NUMBER_OF_READ_PAIRS / 2)), np.concatenate((img_left, colored_disparity), axis=1))
