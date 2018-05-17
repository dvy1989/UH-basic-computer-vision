#! /usr/bin/env python3

# https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html
# https://github.com/abidrahmank/OpenCV2-Python-Tutorials


from os import path, makedirs, getenv
from os.path import sep

import cv2
import numpy as np

OUTPUT_FOLDER = "day4_output"


class KEYPOINT_ALGORITHMS:
    ORB = 0
    SURF = 1
    SIFT = 2


def get_point(keypoints, point_id):
    x, y = keypoints[point_id].pt
    return int(x), int(y)


def make_connected_image_one_side(poster, frame, output_file, algorithm):
    combined, frame_descriptors, frame_keypoints, poster_descriptors, poster_keypoints, poster_width = make_combined_image(frame, poster, algorithm)
    print combined.shape
    frame_nearest_to_poster_points = find_nearest_points(frame_descriptors, poster_descriptors)
    for i in range(0, len(poster_keypoints)):
        if len(poster_keypoints) > i and len(frame_keypoints) > i:
            x1, y1 = get_point(poster_keypoints, i)
            x2, y2 = get_point(frame_keypoints, int(frame_nearest_to_poster_points[i][0]))
            cv2.line(combined, (x1, y1), (x2 + poster_width, y2), thickness=1, color=(0, 0, 255))
    cv2.imwrite(output_file, combined)


def make_connected_image_two_side(poster, frame, output_file, algorithm):
    combined, frame_descriptors, frame_keypoints, poster_descriptors, poster_keypoints, poster_width = make_combined_image(frame, poster, algorithm)
    print combined.shape
    non_matching_points = 0
    frame_nearest_to_poster_points = find_nearest_points(frame_descriptors, poster_descriptors)
    poster_nearest_to_frame_points = find_nearest_points(poster_descriptors, frame_descriptors)
    for i in range(0, len(frame_nearest_to_poster_points)):
        if len(poster_keypoints) > i and len(frame_keypoints) > i:
            poster_point_id = int(poster_nearest_to_frame_points[i][0])
            frame_point_id = int(frame_nearest_to_poster_points[poster_point_id][0])
            if frame_point_id == i:
                x1, y1 = get_point(poster_keypoints, i)
                x2, y2 = get_point(frame_keypoints, int(frame_nearest_to_poster_points[i][0]))
                cv2.line(combined, (x1, y1), (x2 + poster_width, y2), thickness=1, color=(0, 0, 255))
            else:
                non_matching_points += 1
    print "There are %s non-matching points" % non_matching_points
    cv2.imwrite(output_file, combined)


def make_combined_image(frame, poster, algorithm):
    poster_height, poster_width, depth = poster.shape
    poster, poster_keypoints, poster_descriptors = get_keypoints(poster, algorithm)
    frame, frame_keypoints, frame_descriptors = get_keypoints(frame, algorithm)
    poster, frame = pad_images(poster, frame)
    combined = np.concatenate((poster, frame), axis=1)
    return combined, frame_descriptors, frame_keypoints, poster_descriptors, poster_keypoints, poster_width


def pad_images(img1, img2):
    height1, width1, depth = img1.shape
    height2, width2, depth = img2.shape
    if height2 > height1:
        return np.concatenate((img1, np.zeros((max([height1, height2]) - height1, width1, depth))), axis=0), img2
    elif height1 > height2:
        return img1, np.concatenate((img2, np.zeros((max([height1, height2]) - height2, width2, depth))), axis=0)
    return img1, img2


def process_static_images():
    poster = cv2.imread("poster.jpeg")
    frame = cv2.imread("frame.jpeg")
    make_connected_image_one_side(poster, frame, path.join(OUTPUT_FOLDER, "combined_orb_one_side.jpg"), KEYPOINT_ALGORITHMS.ORB)
    make_connected_image_two_side(poster, frame, path.join(OUTPUT_FOLDER, "combined_orb_two_side.jpg"), KEYPOINT_ALGORITHMS.ORB)
    make_connected_image_one_side(poster, frame, path.join(OUTPUT_FOLDER, "combined_surf_one_side.jpg"), KEYPOINT_ALGORITHMS.SURF)
    make_connected_image_two_side(poster, frame, path.join(OUTPUT_FOLDER, "combined_surf_two_side.jpg"), KEYPOINT_ALGORITHMS.SURF)
    make_connected_image_one_side(poster, frame, path.join(OUTPUT_FOLDER, "combined_sift_one_side.jpg"), KEYPOINT_ALGORITHMS.SIFT)
    make_connected_image_two_side(poster, frame, path.join(OUTPUT_FOLDER, "combined_sift_two_side.jpg"), KEYPOINT_ALGORITHMS.SIFT)


def find_nearest_points(descriptors1, descriptors2):
    knn = cv2.ml.KNearest_create()
    knn.train(np.float32(descriptors1), cv2.ml.ROW_SAMPLE, np.arange(len(descriptors1)))
    ret, results, neighbours, dist = knn.findNearest(np.float32(descriptors2), 1)
    return results


def process_video():
    poster = cv2.imread("poster.jpeg")
    video = cv2.VideoCapture("video.avi")
    index = 1
    flag = True
    while flag:
        ret, frame = video.read()
        if frame is not None:
            print ret, frame.shape
            make_connected_image_one_side(poster, frame, path.join(OUTPUT_FOLDER, "%s_combined_orb_one_side.jpg" % index), KEYPOINT_ALGORITHMS.ORB)
            make_connected_image_two_side(poster, frame, path.join(OUTPUT_FOLDER, "%s_combined_orb_two_side.jpg" % index), KEYPOINT_ALGORITHMS.ORB)
            make_connected_image_one_side(poster, frame, path.join(OUTPUT_FOLDER, "%s_combined_surf_one_side.jpg" % index), KEYPOINT_ALGORITHMS.SURF)
            make_connected_image_two_side(poster, frame, path.join(OUTPUT_FOLDER, "%s_combined_surf_two_side.jpg" % index), KEYPOINT_ALGORITHMS.SURF)
            make_connected_image_one_side(poster, frame, path.join(OUTPUT_FOLDER, "%s_combined_sift_one_side.jpg" % index), KEYPOINT_ALGORITHMS.SIFT)
            make_connected_image_two_side(poster, frame, path.join(OUTPUT_FOLDER, "%s_combined_sift_two_side.jpg" % index), KEYPOINT_ALGORITHMS.SIFT)
            index = index + 1
        else:
            flag = False
    video.release()


def hands_on():
    process_static_images()
    process_video()


def get_detector(algorithm):
    if algorithm == KEYPOINT_ALGORITHMS.ORB:
        return cv2.ORB_create()
    elif algorithm == KEYPOINT_ALGORITHMS.SURF:
        return cv2.xfeatures2d.SURF_create(hessianThreshold=1000)
    elif algorithm == KEYPOINT_ALGORITHMS.SIFT:
        return cv2.xfeatures2d.SIFT_create(contrastThreshold=0.07)


def get_keypoints(img, algoritm):
    detector = get_detector(algoritm)
    keypoints, descriptors = detector.detectAndCompute(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
    # cv2.imshow("Test", img_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_with_keypoints, keypoints, descriptors


def pad_image_by_y(img1, img2):
    height1, width1, depth = img1.shape
    height2, width2, depth = img2.shape
    addition1 = np.zeros((max(height1, height2) - height1, width1, depth))
    print img1.shape, addition1.shape
    return np.concatenate((img1, addition1), axis=0), img2


def homework():
    opencv_dir = getenv("OPENCV_DIR")
    cascades_path = path.join(sep.join(opencv_dir.split(sep)[0:-3]), "sources", "data", "haarcascades")
    cascades_path_1 = path.join(sep.join(opencv_dir.split(sep)[0:-3]), "sources", "data", "haarcascades", "cuda")
    face_cascade = cv2.CascadeClassifier(path.join(cascades_path, "haarcascade_frontalface_default.xml"))
    eyes_cascade = cv2.CascadeClassifier(path.join(cascades_path, "haarcascade_eye.xml"))
    img = cv2.imread("people.jpeg")
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_grayscale, 1.7, 5)
    for (x, y, w, h) in faces:
        print x, y, w, h
        face_grayscale = np.array(img_grayscale[y:(y + h), x:(x + w)])
        print face_grayscale.shape
        face_color = img[y:(y + h), x:(x + w)]
        eyes = eyes_cascade.detectMultiScale2(face_grayscale)
        print "%s eyes have been found" % len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if not path.exists(OUTPUT_FOLDER):
        makedirs(OUTPUT_FOLDER)
        # hands_on()
    homework()
