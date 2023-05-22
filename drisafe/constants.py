# constants.py

"""This module defines project-level constants."""

import pathlib

REC = "06"
DS_PATH = pathlib.Path("F:/Veoneer/Datasets/Dr(eye)ve/DREYEVE_DATA")
RT_VID_PATH = DS_PATH / REC / "video_garmin.avi"
ETG_VID_PATH = DS_PATH / REC / "video_etg.avi"
RT_SAMPLE = pathlib.Path().absolute() / "media" / "rt_camera_sample.png"
ETG_SAMPLE = pathlib.Path().absolute() / "media" / "etg_camera_sample.png"

if __name__ == "__main__":
    print(f"{RT_SAMPLE}: {RT_SAMPLE.is_file()}")