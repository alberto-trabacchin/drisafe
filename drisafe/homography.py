import numpy as np
import cv2 as cv
import pathlib
from constants import RT_SAMPLE, ETG_SAMPLE

def read_image(path):
    img = cv.imdecode(np.fromfile(path, dtype = np.uint8), cv.IMREAD_UNCHANGED)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img_gray, img_rgb

def SIFT(img):
    sift_detector = cv.SIFT_create()
    kp, des = sift_detector.detectAndCompute(img, None)
    return kp, des

if __name__ == "__main__":
    img_gray, img_rgb = read_image(RT_SAMPLE)