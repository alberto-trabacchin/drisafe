# constants.py

"""This module defines project-level constants."""

import pathlib

REC = "06"
DS_PATH = pathlib.Path("F:/Veoneer/Datasets/Dr(eye)ve/DREYEVE_DATA")
RT_VID_PATH = DS_PATH / REC / "video_garmin.avi"
ETG_VID_PATH = DS_PATH / REC / "video_etg.avi"
RT_SAMPLE_PATH = pathlib.Path().absolute() / "media" / "rt_camera_sample_2.png"
ETG_SAMPLE_PATH = pathlib.Path().absolute() / "media" / "etg_camera_sample_2.png"

if __name__ == "__main__":
    print(f"{RT_SAMPLE_PATH}: {RT_SAMPLE_PATH.is_file()}")