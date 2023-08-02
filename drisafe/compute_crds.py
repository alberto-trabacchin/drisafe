from drisafe.sstream import SStream
from drisafe import homography
from drisafe.constants import SENSORS
from drisafe.config.paths import COMPUTE_CRDS_LOG_PATH
from multiprocessing import Pool, Process
import cv2 as cv
import numpy as np
import json

def get_empty_array(size):
    arr = np.empty(size)
    arr.fill(np.NaN)
    return arr

def isnan(np_array):
    return np.isnan(np.sum(np_array))

def get_rt_crds(stream, verbose):
    etg_crd = stream.etg_cam.gaze_crd 
    rt_frame_gray = cv.cvtColor(stream.rt_cam.frame, cv.COLOR_BGR2GRAY)
    etg_frame_gray = cv.cvtColor(stream.etg_cam.frame, cv.COLOR_BGR2GRAY)
    rt_kp, rt_des = homography.SIFT(rt_frame_gray)
    etg_kp, etg_des = homography.SIFT(etg_frame_gray)
    matches = homography.match_keypoints(etg_des, rt_des, threshold = homography.KNN_THRESH)
    n_matches = len(matches)
    nan_crds = False
    if isnan(etg_crd):
        rt_crd = get_empty_array(size = (1, 1, 2))
        nan_crds = True
    if (n_matches < 4):
        if verbose:
            print(f"({stream.rec_id}) Not enough matches.")
        rt_crd = get_empty_array(size = (1, 1, 2))
        H = get_empty_array(size = (3, 3))
        mask = get_empty_array(size = (n_matches, 1))
    else:
        H, mask = homography.estimate_homography(etg_kp, rt_kp, matches)
        if H is None:
            if verbose:
                print(f"({stream.rec_id}) Minimization not reached.")
            H = get_empty_array(size = (3, 3))
            mask = get_empty_array(size = (n_matches, 1))
            rt_crd = get_empty_array(size = (1, 1, 2))
        elif not nan_crds:
            rt_crd = homography.project_gaze(etg_crd, H)
    num_rt_kp = len(rt_kp)
    num_etg_kp = len(etg_kp)
    return rt_crd, H, num_rt_kp, num_etg_kp, n_matches

def save_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')

def write_data(data, id):
    data_path = SENSORS[id - 1]["gaze_track"]["rt_crd_path"]
    with open(data_path, 'w') as fl:
        json.dump(data, fl, default = save_default, indent = 2)
    print(f"({id}) Data written.")

def worker(args):
    rec_id, verbose = args
    print(f"Running process {rec_id}.")
    stream = SStream(rec_id)
    data_list = []
    while True:
        stream.read()
        if not stream.online: break
        rt_crd, H, num_rt_kp, num_etg_kp, n_matches = get_rt_crds(stream, verbose)
        data_list.append({
            "rt_crd": rt_crd.reshape(2),
            "etg_crd": stream.etg_cam.gaze_crd.reshape(2),
            "H": H,
            "num_rt_kp": num_rt_kp,
            "num_etg_kp": num_etg_kp,
            "n_matches": n_matches
        })
        stream.rt_cam.gaze_crd = rt_crd
        if verbose:
            stream.show_coordinates()

    stream.close()
    write_data(data_list, rec_id)
    return True
    

if __name__ == "__main__":
    rec_ids = range(1, 75)
    p = Pool(processes = 10)
    verbose = False
    args = [(id, verbose) for id in rec_ids]
    results = p.map(worker, args)
    p.close()
    p.join()
    print(results)
