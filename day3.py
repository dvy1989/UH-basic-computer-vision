#! /usr/bin/env python3

# https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html
# https://github.com/abidrahmank/OpenCV2-Python-Tutorials


import cv2
import numpy as np


def get_initial_point(img, height, width):
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            pixel_scale = img[i, j]
            if pixel_scale > 0:
                return i, j


def find_tb_pixels(img):
    tb_pixels = list()
    height, width = img.shape
    tmp = img.copy()
    tmp[tmp > 0] = 1
    x0, y0 = get_initial_point(tmp, height, width)
    visit_pixel(x0, y0, tmp, None, height, width)
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            if tmp[i, j] > 2 or tmp[i, j] == 1:
                tb_pixels.append((j, i))
    return tb_pixels


def visit_pixel(x, y, img, visited_pixels, height, width):
    if visited_pixels is not None and (x, y) in visited_pixels or img[x, y] == 0:
        return False
    if visited_pixels is None:
        visited_pixels = [(x, y)]
    else:
        visited_pixels.append((x, y))
    path_num = 1
    for i in range(-1, 2):
        for j in range(-1, 2):
            x1 = x - i
            y1 = y - j
            if 0 <= x1 < height:
                if 0 <= y1 < width:
                    if visit_pixel(x1, y1, img, visited_pixels, height, width):
                        path_num += 1
    img[x, y] = path_num
    return True


def check_branch(pixel_neighbours):
    if as_int(pixel_neighbours[0, 0:3]) % 3 != 0:
        if as_int(pixel_neighbours[2, 0:3]) % 3 != 0:
            if as_int(pixel_neighbours[0:3, 0]) % 3 != 0:
                if as_int(pixel_neighbours[0:3, 2]) % 3 != 0:
                    return True
    return False


def as_int(bit_arr):
    result = 0
    for bit in np.clip(bit_arr, 0, 1):
        result = (result << 1) | bit
    return result


def draw_skeleton(thinngig_type, str_thinning_type):
    bear_skeleton = cv2.ximgproc.thinning(bear_image, thinningType=thinngig_type)
    cv2.imshow("Bear thinning %s" % str_thinning_type, bear_skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("skeleton_%s.jpg" % str_thinning_type, bear_skeleton)
    tb_pixels = find_tb_pixels(bear_skeleton)
    bear_colored = cv2.cvtColor(bear_skeleton, cv2.COLOR_GRAY2BGR)
    for tb_pixel in tb_pixels:
        cv2.circle(bear_colored, tb_pixel, radius=5, color=(0, 0, 255), thickness=2)
    cv2.imshow("Bear pixels %s" % str_thinning_type, bear_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("pixels_%s.jpg" % str_thinning_type, bear_colored)


if __name__ == "__main__":
    print("Reading the image")
    bear_image = cv2.imread("junction.png", cv2.IMREAD_GRAYSCALE)
    # bear_image = cv2.imread("bear.pbm", cv2.IMREAD_GRAYSCALE)

    bear_distance_transformed = cv2.distanceTransform(np.uint8(bear_image / 255.0), distanceType=cv2.DIST_L1, maskSize=5)
    cv2.imshow("Bear", bear_distance_transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    draw_skeleton(cv2.ximgproc.THINNING_ZHANGSUEN, "ZHANGSUEN")
    draw_skeleton(cv2.ximgproc.THINNING_GUOHALL, "GUOHALL")
