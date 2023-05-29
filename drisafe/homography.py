import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from drisafe.constants import RT_SAMPLE_PATH, ETG_SAMPLE_PATH

FLANN_INDEX_KDTREE = 1
RANSAC_THRESH = 5
RANSAC_MAX_ITERS = 2000
KNN_THRESH = 0.5

def read_image(path):
    img_bgr = cv.imdecode(np.fromfile(path, dtype = np.uint8), cv.IMREAD_UNCHANGED)
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    return img_gray, img_bgr

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
    img_matches = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)
    plt.imshow(img_matches)
    plt.show()

def draw_gaze(img, coord):
    coord = coord.astype(np.int32)
    img = cv.circle(img, coord, radius = 25, thickness = 2, color = (255, 0, 0))
    img = cv.circle(img, coord, radius = 6, thickness = -1, color = (255, 0, 0))
    return img

def print_gaze(rt_img, etg_img, rt_coords, etg_coords):
    rt_img = cv.cvtColor(rt_img, cv.COLOR_BGR2RGB)
    etg_img = cv.cvtColor(etg_img, cv.COLOR_BGR2RGB)
    for rt_c, etg_c in zip(rt_coords, etg_coords):
        rt_img = draw_gaze(rt_img, rt_c)
        etg_img = draw_gaze(etg_img, etg_c)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rt_img)
    ax[1].imshow(etg_img)
    plt.show()

def mesh_gaze_coords(nx, ny, img):
    width = img.shape[1]
    height = img.shape[0]
    x = np.linspace(0, width, nx, dtype = np.int32)
    y = np.linspace(0, height, ny, dtype = np.int32)
    xv, yv = np.meshgrid(x, y, indexing = "xy")
    pts = np.dstack((xv, yv)).reshape(-1, 2)
    return pts

def estimate_homography(kp1, kp2, matches, verbose = False):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, RANSAC_THRESH, maxIters = RANSAC_MAX_ITERS)
    if verbose:
        print(f"(KNN thresh: {KNN_THRESH}, RANSAC thresh: {RANSAC_THRESH}) - " \
              f"{np.count_nonzero(mask)} keypoints detected.")
    return H, mask

def project_gaze(etg_coords, H):
    etg_coords = etg_coords.reshape(1, -1, 2).astype(np.float32)
    rt_coords = cv.perspectiveTransform(etg_coords, H).reshape(-1, 2)
    return rt_coords

def show_sift_kp_imgs(img1, img2):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    fig.suptitle("Keypoints detection", fontsize = 14)
    plt.show()

if __name__ == "__main__":
    rt_img_gray, rt_img_bgr = read_image(RT_SAMPLE_PATH)
    etg_img_gray, etg_img_bgr = read_image(ETG_SAMPLE_PATH)
    rt_kp, rt_des = SIFT(rt_img_gray)
    etg_kp, etg_des = SIFT(etg_img_gray)
    matches = match_keypoints(etg_des, rt_des, threshold = KNN_THRESH)
    H, mask = estimate_homography(etg_kp, rt_kp, matches, verbose = True)
    #etg_coords = np.array([[791, 464]])
    etg_coords = mesh_gaze_coords(nx = 5, ny = 5, img = etg_img_gray)
    rt_coords = project_gaze(etg_coords, H)
    plot_matches(matches, etg_kp, rt_kp, etg_img_bgr, rt_img_bgr, mask)
    print_gaze(etg_img_bgr, rt_img_bgr, etg_coords, rt_coords)