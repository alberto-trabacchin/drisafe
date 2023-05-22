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

def match_kps(des1, des2, threshold):
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(des1, des2, k = 2)
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    return good_matches

def plot_sift(img_gray, img_rgb, kp):
    tmp = img_rgb.copy()
    out_img = cv.drawKeypoints(img_gray, kp, tmp, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out_img

def plot_matches(matches, kp1, kp2, img1, img2):
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), 
                            img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    draw_params = dict(matchColor = (-1),
                       singlePointColor = None,
                       flags = 2)
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img_matches)
    plt.show()

def estimate_homography(kp1, kp2, matches):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    return M, mask

def compose_total_img(img1, img2):
    height = max(img1.shape[0], img2.shape[0])
    ar1 = img1.shape[1] / img1.shape[0]
    ar2 = img2.shape[1] / img2.shape[0]
    img1 = cv.resize(img1, dsize = (int(ar1 * height), height))
    img2 = cv.resize(img2, dsize = (int(ar2 * height), height))
    total_img = np.concatenate((img1, img2), axis = 1)
    return total_img

def show_sift_kp_imgs(img1, img2):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    fig.suptitle("Keypoints detection", fontsize = 14)
    plt.show()

if __name__ == "__main__":
    rt_img_gray, rt_img_rgb = read_image(RT_SAMPLE_PATH)
    etg_img_gray, etg_img_rgb = read_image(ETG_SAMPLE_PATH)
    rt_kp, rt_des = SIFT(rt_img_gray)
    etg_kp, etg_des = SIFT(etg_img_gray)
    matches = match_kps(rt_des, etg_des, threshold = 0.5)
    M, mask = estimate_homography(rt_kp, etg_kp, matches)