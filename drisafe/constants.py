# constants.py

"""This module defines project-level constants."""

import pathlib

DS_PATH = pathlib.Path("F:/Veoneer/Datasets/Dr(eye)ve/DREYEVE_DATA")
_recordings = [d for d in list(DS_PATH.iterdir()) if d.is_dir()]
RT_VID_PATHS = [s / "video_garmin.avi" for s in _recordings]
ETG_VID_PATHS = [s / "video_etg.avi" for s in _recordings]
ETG_DATA_PATHS = [s / "etg_samples.txt" for s in _recordings]
GPS_PATHS = [s / "speed_course_coord.txt" for s in _recordings]
RT_SAMPLE_PATH = pathlib.Path().absolute() / "media" / "rt_camera_sample_5.png"
ETG_SAMPLE_PATH = pathlib.Path().absolute() / "media" / "etg_camera_sample_5.png"

FPS_RT = 25
FPS_ETG = 30

SENSORS = {
    "gaze_track": {
        "name": "Gaze Tracker",
        "paths": ETG_DATA_PATHS
    },
    "etg_cam": {
        "name": "ETG Camera",
        "paths": ETG_VID_PATHS,
        "fps": FPS_ETG
    },
    "roof_cam": {
        "name": "Roof Top Camera",
        "paths": RT_VID_PATHS,
        "fps": FPS_RT
    }
}

def _check_all(path_list, dtype = "file"):
    if dtype == "file":
        return all([f.is_file() for f in path_list])
    elif dtype == "folder":
        return all([f.is_dir() for f in path_list])
    else:
        return False

if __name__ == "__main__":
    print(f"DS main dir: {DS_PATH.is_dir()}")
    print(f"Samples: {_check_all(_recordings, 'folder')}")
    print(f"RT videos: {_check_all(RT_VID_PATHS, 'file')}")
    print(f"ETG videos: {_check_all(ETG_VID_PATHS, 'file')}")
    print(f"ETG data: {_check_all(ETG_DATA_PATHS, 'file')}")
    print(f"GPS data: {_check_all(GPS_PATHS, 'file')}")
    print(f"RT image sample: {_check_all([RT_SAMPLE_PATH], 'file')}")
    print(f"ETG image sample: {_check_all([ETG_SAMPLE_PATH], 'file')}")