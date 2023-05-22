import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
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
    rt_img_gray, rt_img_rgb = read_image(RT_SAMPLE)
    etg_img_gray, etg_img_rgb = read_image(ETG_SAMPLE)
    rt_kp, rt_des = SIFT(rt_img_gray)
    etg_kp, etg_des = SIFT(etg_img_gray)
    rt_kp_img = plot_sift(rt_img_gray, rt_img_rgb, rt_kp)
    etg_kp_img = plot_sift(etg_img_gray, etg_img_rgb, etg_kp)
    show_sift_kp_imgs(rt_kp_img, etg_kp_img)