from drisafe.sensorstreams import SensorStreams
from drisafe.constants import SENSORS
from drisafe.config.paths import _recordings
from drisafe import homography
import cv2 as cv
import pandas as pd
import numpy as np
from multiprocessing import Process


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

def writer(rec_id):
    print(f"Ran process {rec_id}.")
    sen_stream = SensorStreams(SENSORS[rec_id], rec_id)
    rt_crd_path = sen_stream.etg_tracker["rt_crd_path"]
    ds_rt_crd = pd.DataFrame({"X": [], "Y": []})
    while True:
        sen_stream.read(show_gaze_crd = True)
        if not sen_stream.online: break
        rt_crds = calc_rt_crds(sen_stream)
        print(f"({sen_stream.rec_id}-{sen_stream.t_step}) - RT gaze: {rt_crds}")
        [[[x, y]]] = rt_crds
        ds_rt_crd = pd.concat([ds_rt_crd, pd.DataFrame([{"X": x, "Y": y}])], ignore_index=True)
        #homography.print_gaze(sen_stream.rt_frame, sen_stream.etg_frame, rt_crds, sen_stream.etg_crd)
        #if (cv.waitKey(1) & 0xFF == ord("q")):
        #    break
    ds_rt_crd.to_csv(rt_crd_path)
    print("Data saved.")
    
if __name__ == "__main__":
    rec_ids1 = range(0, 25)
    rec_ids2 = range(25, 50)
    rec_ids3 = range(50, 74)
    missing_ids = [52, 53, 55, 56, 57, 60, 61, 62, 63, 66, 67, 69, 71, 72]
    procs = []
    for id in rec_ids1:
        p = Process(target = writer, args = (id,))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
        