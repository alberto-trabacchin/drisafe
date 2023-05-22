import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from constants import RT_SAMPLE_PATH, ETG_SAMPLE_PATH

def read_image(path):
    img = cv.imdecode(np.fromfile(path, dtype = np.uint8), cv.IMREAD_UNCHANGED)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img_gray, img_rgb

def SIFT(img):
    sift_detector = cv.SIFT_create()
    kp, des = sift_detector.detectAndCompute(img, None)
    return kp, des

def match_kps(kp1, des1, kp2, des2, threshold):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)
    goods = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            goods.append([m])
    matches = []
    for pair in goods:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))
    matches = np.array(matches)
    return matches

def plot_sift(img_gray, img_rgb, kp):
    tmp = img_rgb.copy()
    out_img = cv.drawKeypoints(img_gray, kp, tmp, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out_img

def show_sift_kp_imgs(img1, img2):
    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(img1)
    axis[1].imshow(img2)
    fig.suptitle("Keypoints detection", fontsize = 14)
    plt.show()

if __name__ == "__main__":
    rt_img_gray, rt_img_rgb = read_image(RT_SAMPLE_PATH)
    etg_img_gray, etg_img_rgb = read_image(ETG_SAMPLE_PATH)
    rt_kp, rt_des = SIFT(rt_img_gray)
    etg_kp, etg_des = SIFT(etg_img_gray)
    matches = match_kps(rt_kp, rt_des, etg_kp, etg_des, threshold = 0.5)