from drisafe.sensorstreams import SensorStreams
from drisafe.constants import SENSORS
from drisafe.config.paths import _recordings
from drisafe import homography
import cv2 as cv
import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue


def calc_rt_crds(sen_stream):
    rt_frame_gray = cv.cvtColor(sen_stream.rt_frame, cv.COLOR_BGR2GRAY)
    etg_frame_gray = cv.cvtColor(sen_stream.etg_frame, cv.COLOR_BGR2GRAY)
    rt_kp, rt_des = homography.SIFT(rt_frame_gray)
    etg_kp, etg_des = homography.SIFT(etg_frame_gray)
    matches = homography.match_keypoints(etg_des, rt_des, threshold = homography.KNN_THRESH)
    if len(matches) < 4:
        rt_crds = write_empty_crds("Not enough matches.")
    else:
        H, mask = homography.estimate_homography(etg_kp, rt_kp, matches)
        if H is None:
            rt_crds = write_empty_crds("Minimization not reached.")
        else:
            rt_crds = homography.project_gaze(sen_stream.etg_crd, H)
    return rt_crds

def write_empty_crds(message):
    rt_crds = np.empty((1, 1, 2))
    rt_crds[:] = np.nan
    print(f"{message}. Written NaN.")
    return rt_crds

def save_rt_crds(data):
    rt_crds = data["rt_crds"]
    rt_crds.to_csv(data["path"])
    print(f"Recording {data['rec_id']} saved.")

def worker(id, queue):
    print(f"Ran process {id}.")
    sen_stream = SensorStreams(SENSORS[id - 1], id)
    rt_crd_path = sen_stream.etg_tracker["rt_crd_path"]
    ds_rt_crd = pd.DataFrame({"X": [], "Y": []})
    while True:
        sen_stream.read(show_gaze_crd = True)
        if not sen_stream.online: break
        rt_crds = calc_rt_crds(sen_stream)
        print(f"({sen_stream.rec_id}-{sen_stream.t_step}) - RT gaze: {rt_crds}")
        [[[x, y]]] = rt_crds
        ds_rt_crd = pd.concat([ds_rt_crd, pd.DataFrame([{"X": x, "Y": y}])], ignore_index=True)
    queue.put({
        "rec_id": id,
        "rt_crds": ds_rt_crd,
        "path": rt_crd_path
    })
    
if __name__ == "__main__":
    rec_ids = [60, 66, 69]
    queue = Queue()
    jobs = []
    for id in rec_ids:
        p = Process(target = worker, args = (id, queue))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    while not queue.empty():
        save_rt_crds(data = queue.get())