import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from constants import ETG_VID_PATH, RT_VID_PATH, FPS_RT, FPS_ETG

def open_camera(cam_path):
    cap = cv.VideoCapture(str(cam_path))
    if not cap.isOpened():
        print(f"Error opening video stream {cam_path}.")
    return cap

def init_cams_params(etg_cap, rt_cap):
    time = 0
    k = 0
    if not (etg_cap.isOpened() & rt_cap.isOpened()):
        return
    etg_ret, etg_frame = etg_cap.read()
    rt_ret, rt_frame = rt_cap.read()
    return etg_ret, etg_frame, rt_ret, rt_frame, k, time

def read_frames(rt_cap, rt_ret, rt_frame, etg_cap, k, time):
    etg_ret, etg_frame = etg_cap.read()
    time += 1 / float(FPS_ETG)
    if (k / float(FPS_RT) < time):
        k += 1
        rt_ret, rt_frame = rt_cap.read()
    return etg_ret, etg_frame, rt_ret, rt_frame, k, time

def read_cameras(etg_cam_path, rt_cam_path):
    etg_cap = open_camera(etg_cam_path)
    rt_cap = open_camera(rt_cam_path)
    etg_ret, etg_frame, rt_ret, rt_frame, k, time = init_cams_params(etg_cap, rt_cap)
    while (etg_cap.isOpened() and rt_cap.isOpened()):
        etg_ret, etg_frame, rt_ret, rt_frame, k, time = read_frames(rt_cap, rt_ret, rt_frame,
                                                                    etg_cap, k, time)
        if not (etg_ret and rt_ret):
            break
        # Do all stuffs with frames...
        cv.imshow("ETG", etg_frame)
        cv.imshow("RT", rt_frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    etg_cap.release()
    rt_cap.release()


if __name__ == "__main__":
    read_cameras(ETG_VID_PATH, RT_VID_PATH)