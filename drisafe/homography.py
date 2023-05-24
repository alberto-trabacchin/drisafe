import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from constants import RT_SAMPLE_PATH, ETG_SAMPLE_PATH

FLANN_INDEX_KDTREE = 1

def read_image(path):
    img = cv.imdecode(np.fromfile(path, dtype = np.uint8), cv.IMREAD_UNCHANGED)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img_gray, img_rgb

def SIFT(img):
    sift_detector = cv.SIFT_create()
    kp, des = sift_detector.detectAndCompute(img, None)
    return kp, des

def match_keypoints(des1, des2, threshold):
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k = 2)   # For each kp in des 1 find the k-th (k=2) nn in des2
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:     # If the distance btw. the two possible matches is high, take the first match as good match
            good_matches.append(m)
    return good_matches

def plot_sift(img_gray, img_rgb, kp):
    tmp = img_rgb.copy()
    out_img = cv.drawKeypoints(img_gray, kp, tmp, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out_img

def plot_matches(matches, kp1, kp2, img1, img2, mask = None):
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), 
                            img1.shape[1] + img2.shape[1], 3), 
                            dtype = np.uint8)
    if mask is not None:
        mask = mask.ravel().tolist()
    draw_params = dict(matchColor = (-1),
                       singlePointColor = None,
                       matchesMask = mask,
                       flags = 2)
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img_matches)
    plt.show()

def draw_gaze(img, coord):
    coord = coord.reshape(2).astype(np.int32)
    img = cv.circle(img, coord, radius = 25, thickness = 2, color = (255, 0, 0))
    img = cv.circle(img, coord, radius = 6, thickness = -1, color = (255, 0, 0))
    return img

def print_gazes(img1, img2):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    plt.show()

def estimate_homography(kp1, kp2, matches):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    return H, mask

def project_gaze(etg_coords, H):
    etg_coords = etg_coords.reshape(1, -1, 2).astype(np.float32)
    rt_coords = cv.perspectiveTransform(etg_coords, H)
    return rt_coords

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
    matches = match_keypoints(rt_des, etg_des, threshold = 0.5)
    H, mask = estimate_homography(rt_kp, etg_kp, matches)
    print(np.count_nonzero(mask))
    etg_coords = np.array([[600, 600],
                           [600, 800]])
    rt_coords = project_gaze(etg_coords, H)
    #plot_matches(matches, rt_kp, etg_kp, rt_img_rgb, etg_img_rgb, mask)
