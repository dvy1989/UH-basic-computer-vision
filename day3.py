#! /usr/bin/env python3

# https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html
# https://github.com/abidrahmank/OpenCV2-Python-Tutorials


import cv2
import numpy as np


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
    bear_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bear_distance_transformed = cv2.distanceTransform(bear_image, distanceType=cv2.DIST_L2, maskSize=0)
    cv2.imshow(name, bear_distance_transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    draw_skeleton(name, bear_image, cv2.ximgproc.THINNING_ZHANGSUEN, "ZHANGSUEN")
    draw_skeleton(name, bear_image, cv2.ximgproc.THINNING_GUOHALL, "GUOHALL")


if __name__ == "__main__":
    print("Reading the image")
    # bear_image = cv2.imread("junction.png", cv2.IMREAD_GRAYSCALE)
    # bear_image = cv2.imread("bear.pbm", cv2.IMREAD_GRAYSCALE)
    process_image("car.bmp", "Car")
