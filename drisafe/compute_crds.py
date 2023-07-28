from drisafe.sstream import SStream
from drisafe import homography
from drisafe.constants import SENSORS
from multiprocessing import Pool
import cv2 as cv
import numpy as np
import json
import itertools

def get_empty_array(size):
    arr = np.empty(size)
    arr.fill(np.NaN)
    return arr

def isnan(np_array):
    return np.isnan(np.sum(np_array))

def get_rt_crds(stream):
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
        print("Not enough matches.")
        H = get_empty_array(size = (3, 3))
        mask = get_empty_array(size = (n_matches, 1))
    else:
        H, mask = homography.estimate_homography(etg_kp, rt_kp, matches)
        if H is None:
            print("Minimization not reached.")
            H = get_empty_array(size = (3, 3))
            mask = get_empty_array(size = (n_matches, 1))
            if not nan_crds:
                rt_crd = get_empty_array(size = (1, 1, 2))
        elif H is not None and not nan_crds:
            rt_crd = homography.project_gaze(etg_crd, H)
    return rt_crd, H, mask

def worker(rec_id):
    print(f"Running process {rec_id}.")
    stream = SStream(rec_id)
    data = {
        "id": rec_id,
        "rt_crd": [],
        "H": [],
        "mask": []
    }
    while stream.online:
        stream.read()
        rt_crd, H, mask  = get_rt_crds(stream)
        data["rt_crd"].append(rt_crd)
        data["H"].append(H)
        data["mask"].append(mask)
        print(f"({rec_id}, {stream.t_step}) ETG: {stream.etg_cam.gaze_crd}")
        print(f"({rec_id}, {stream.t_step}) RT: {rt_crd}")
        # Temporary
        if stream.t_step == 10: break
    stream.close()
    return data

def run_workers(rec_ids, max_workers = 10):
    results = []
    n = max_workers
    groups = [rec_ids[i : i + n] for i in range(0, len(rec_ids), n)]
    for rec_ids in groups:
        pool = Pool()
        result = pool.map(worker, rec_ids)
        pool.close()
        pool.join()
        results.append(result)
    merged_results = list(itertools.chain(*results))
    return merged_results

def save_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')

def write_data(results):
    for res in results:
        id = res["id"]
        data_path = SENSORS[id - 1]["gaze_track"]["rt_crd_path"]
        with open(data_path, 'w') as fl:
            json.dump(res, fl, default = save_default)
        
    
if __name__ == "__main__":
    rec_ids = range(1, 2)
    results = run_workers(rec_ids, max_workers = 10)
    print([r["id"] for r in results])
    write_data(results)
