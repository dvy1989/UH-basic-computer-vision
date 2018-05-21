#! /usr/bin/env python3

# https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html
# https://github.com/abidrahmank/OpenCV2-Python-Tutorials
import gzip
import pickle
from os import path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import cv2


def get_initial_point(img, height, width):
    for x in range(0, height - 1):
        for y in range(0, width - 1):
            non_zero_pixels = 0
            pixel_scale = img[x, y]
            if pixel_scale > 0:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        x1 = x + i
                        y1 = y + j
                        if 0 <= x1 < height:
                            if 0 <= y1 < width:
                                if img[x1, y1] > 0:
                                    if x1 != x or y1 != y:
                                        non_zero_pixels += 1
                if non_zero_pixels == 1:
                    return x, y
    return 0, 0


def find_edges(img, x, y, currentPoint, edges, height, width, visited_pixels):
    if (x, y) in visited_pixels:
        return
    visited_pixels.append((x, y))
    if (img[x, y] > 2 or img[x, y] == 1) and (y, x) != currentPoint:
        edges.add((currentPoint, (y, x)))
        newPoint = (y, x)
    else:
        newPoint = currentPoint
    for i in range(-1, 2):
        for j in range(-1, 2):
            x1 = x + i
            y1 = y + j
            if 0 <= x1 < height:
                if 0 <= y1 < width:
                    if img[x1, y1] > 0:
                        find_edges(img, x1, y1, newPoint, edges, height, width, visited_pixels)


def find_tb_pixels(img):
    height, width = img.shape
    tmp = img.copy()
    tmp[tmp > 0] = 1
    x0, y0 = get_initial_point(tmp, height, width)
    # tmp[x0, y0] = -1
    visit_pixel(x0, y0, tmp, list(), height, width)
    edges = set()
    find_edges(tmp, x0, y0, (y0, x0), edges, height, width, list())
    return edges


def visit_pixel(x, y, img, visited_pixels, height, width):
    pixels_to_visit = list()
    non_zero_pixels = 0;
    for i in range(-1, 2):
        for j in range(-1, 2):
            x1 = x + i
            y1 = y + j
            if 0 <= x1 < height:
                if 0 <= y1 < width:
                    if img[x1, y1] > 0:
                        if x1 != x or y1 != y:
                            non_zero_pixels += 1
                        if (x1, y1) not in visited_pixels:
                            visited_pixels.append((x1, y1))
                            pixels_to_visit.append((x1, y1))
    if len(pixels_to_visit) > 0:
        for x1, y1 in pixels_to_visit:
            if visit_pixel(x1, y1, img, visited_pixels, height, width):
                img[x, y] += 1
    else:
        return non_zero_pixels == 1
    return True


def draw_skeleton(name, bear_image, thinngig_type, str_thinning_type):
    bear_skeleton = cv2.ximgproc.thinning(bear_image, thinningType=thinngig_type)
    cv2.imshow("%s thinning %s" % (name, str_thinning_type), bear_skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("%s_skeleton_%s.jpg" % (name, str_thinning_type), bear_skeleton)
    tb_pixels = find_tb_pixels(bear_skeleton)
    bear_colored = cv2.cvtColor(bear_skeleton, cv2.COLOR_GRAY2BGR)
    processed_points = set()
    for point1, point2 in tb_pixels:
        if point1 not in processed_points:
            cv2.circle(bear_colored, point1, 5, thickness=2, color=(0, 0, 255))
            processed_points.add(point1)
        if point2 not in processed_points:
            cv2.circle(bear_colored, point2, 5, thickness=2, color=(0, 0, 255))
            processed_points.add(point2)
        cv2.line(bear_colored, point1, point2, color=(0, 0, 255), thickness=2)
    cv2.imshow("%s pixels %s" % (name, str_thinning_type), bear_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("%s_pixels_%s.jpg" % (name, str_thinning_type), bear_colored)


def process_image(path, name):
    img_to_process = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img_canny = cv2.Canny(img_to_process, 10, 100)
    # cv2.imshow("Test", img_canny)
    ret, binary_img = cv2.threshold(img_to_process, 100, 255, cv2.THRESH_BINARY)
    bear_distance_transformed = cv2.distanceTransform(binary_img, distanceType=cv2.DIST_L1, maskSize=5)
    cv2.imshow(name, bear_distance_transformed)
    # cv2.imwrite("%s.jpg" % name, bear_distance_transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    draw_skeleton(name, img_to_process, cv2.ximgproc.THINNING_ZHANGSUEN, "ZHANGSUEN")
    draw_skeleton(name, img_to_process, cv2.ximgproc.THINNING_GUOHALL, "GUOHALL")


COEFFS = [2 ** i for i in range(0, 8)]


def calculate_hist(img):
    hist = dict()
    height, width = img.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            sum = int(img[i - 1, j - 1] * COEFFS[0] + img[i - 1, j] * COEFFS[1] + img[i - 1, j + 1] * COEFFS[2] + img[
                i, j + 1] * COEFFS[3] + img[i + 1, j + 1] * COEFFS[4] + img[i + 1, j] * COEFFS[5] + img[i + 1, j - 1] * \
                  COEFFS[6] + img[i, j - 1] * COEFFS[7])
            if sum in hist:
                hist[sum] += 1
            else:
                hist[sum] = 1
    return hist


def homework():
    with gzip.open(path.join("mnist.pkl.gz"), "rb") as f:
        train_set, valid_set, test_set = pickle.load(f)
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set

    features = list()

    for picture in train_x[0:1000]:
        img = picture.reshape(28, 28)
        print cv2.calcHist(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), channels=[0,1,2], maks=)
        cv2.imshow("Pict", img)
        cv2.waitKey(5) & 0xFF
        features.append(calculate_hist(img))

    test_features = list()

    for picture in test_x[0:2000]:
        img = picture.reshape(28, 28)
        # cv2.imshow("Pict", img)
        # cv2.waitKey(5) & 0xFF
        test_features.append(calculate_hist(img))

    print(features[0])

    knn = cv2.ml.KNearest_create();
    knn.train(features, cv2.ml.ROW_SAMPLE, range(0, 1000))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Reading the image")
    # bear_image = cv2.imread("junction.png", cv2.IMREAD_GRAYSCALE)
    # bear_image = cv2.imread("bear.pbm", cv2.IMREAD_GRAYSCALE)
    # for img, name in [("bear.pbm", "Bear"), ("car.bmp", "Car"), ("star.bmp", "Star")]:
    #     process_image(img, name)
    homework()
