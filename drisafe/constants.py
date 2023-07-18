# constants.py

"""This module defines project-level constants."""

from drisafe.config import paths

FPS_RT = 25
FPS_ETG = 30

DOWNTOWN_REC_IDS = [4, 6, 7, 10, 11, 12, 13, 16, 18, 19, 26, 27, 35, 38, 39, 40, 47, 51, 53, 58, 60, 61, 64, 65, 70, 72]

SENSORS = [
    {
        "gaze_track": {
        "name": "Gaze Tracker",
        "etg_crd_path": paths.ETG_DATA_PATHS[i],
        "rt_crd_path": paths.ETG_PROJ_DATA_PATHS[i]
        },
        "etg_cam": {
        "name": "ETG Camera",
        "vid_path": paths.ETG_VID_PATHS[i],
        "fps": FPS_ETG
        },
        "roof_cam": {
        "name": "Roof Top Camera",
        "vid_path": paths.RT_VID_PATHS[i],
        "fps": FPS_RT
        }
    }
    for i in range(len(paths._recordings))]

if __name__ == "__main__":
    print(SENSORS[0])