from drisafe.sensorstreams import SensorStreams
from drisafe.constants import SENSORS
from drisafe.constants import _recordings
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
    if not (len(rt_kp) >= 4 and len(etg_kp) >= 4):
        rt_crds = np.empty((1, 2))
        rt_crds[:] = np.nan
        print("Written NaN.")
    else:
        matches = homography.match_keypoints(etg_des, rt_des, threshold = homography.KNN_THRESH)
        H, mask = homography.estimate_homography(etg_kp, rt_kp, matches)
        rt_crds = homography.project_gaze(sen_stream.etg_crd, H)
    return rt_crds.reshape(2)

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
        [x, y] = rt_crds
        ds_rt_crd = pd.concat([ds_rt_crd, pd.DataFrame([{"X": x, "Y": y}])], ignore_index=True)
        #homography.print_gaze(sen_stream.rt_frame, sen_stream.etg_frame, rt_crds, sen_stream.etg_crd)
        #if (cv.waitKey(1) & 0xFF == ord("q")):
        #    break
    ds_rt_crd.to_csv(rt_crd_path)
    print("Data saved.")
    
if __name__ == "__main__":
    NUM_PROCS = 37
    rec_ids = range(NUM_PROCS)
    #rec_ids = [1]
    procs = []
    for id in rec_ids:
        p = Process(target = writer, args = (id,))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
        