#! /usr/bin/env python3

# https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html
# https://github.com/abidrahmank/OpenCV2-Python-Tutorials


import cv2
import numpy as np

def get_keypoints(img):
    orb_detector = cv2.ORB_create() 
    keypoints, descriptors = orb_detector.detectAndCompute(img, None)
    poster_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
    # cv2.imshow("Test", poster_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return poster_with_keypoints, keypoints, descriptors

def get_keypoints_surf(img):
    surf_detector = cv2.xfeatures2d.SURF_create(1000) 
    keypoints, descriptors = surf_detector.detectAndCompute(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
    # cv2.imshow("Test", poster_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_with_keypoints, keypoints, descriptors

def pad_image_by_y(img1, img2):
    height1, width1, depth = img1.shape
    height2, width2, depth = img2.shape
    addition1 = np.zeros((max(height1, height2) - height1, width1, depth))
    print img1.shape, addition1.shape
    return np.concatenate((img1, addition1), axis=0), img2

def make_a_combined_image(poster, frame, output_image_path):
    img1, keypoints1, descriptors1 = get_keypoints_surf(poster)
    img2, keypoints2, descriptors2 = get_keypoints_surf(frame)
    height1, width1, depth = img1.shape
    print img1.shape, img2.shape
    img1, img2 = pad_image_by_y(poster, frame)
    print img1.shape, img2.shape                
    combined_img = np.concatenate((img1, img2), axis=1)
    knn1 = cv2.ml.KNearest_create()
    knn1.train(np.float32(descriptors1), cv2.ml.ROW_SAMPLE, np.arange(len(descriptors1)))    
    ret12, results12, neighbours12, dist12 = knn1.findNearest(np.float32(descriptors2), 1)
    knn2 = cv2.ml.KNearest_create()
    knn2.train(np.float32(descriptors2), cv2.ml.ROW_SAMPLE, np.arange(len(descriptors2)))    
    ret21, results21, neighbours21, dist21 = knn2.findNearest(np.float32(descriptors1), 1)
    non_matched = 0
    for i in range(0, len(results12)):
        p1 = int(results12[i][0])
        p2 = int(results21[p1][0])
        if i == p2:
            x1, y1 = keypoints1[p1].pt
            x2, y2 = keypoints2[p1].pt
            cv2.line(combined_img, (int(x1),int(y1)), (int(x2)+width1,int(y2)), color=(0, 0, 255), thickness=1)
        else:
            non_matched += 1
    print "There are %s non-matching points" % non_matched
    cv2.imwrite(output_image_path,combined_img)

if __name__ == "__main__":
    poster = cv2.imread("poster.jpeg")
    video = cv2.VideoCapture("video.avi")
    index = 1
    flag = True
    while flag:
        ret, frame = video.read()
        if frame is not None:
            print ret, frame.shape
            make_a_combined_image(poster, frame, "comb%s.jpg" % index)
            index += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            flag = False
    video.release()
    
