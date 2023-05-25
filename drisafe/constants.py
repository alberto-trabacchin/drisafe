# constants.py

"""This module defines project-level constants."""

import pathlib

REC = "06"
DS_PATH = pathlib.Path("F:/Veoneer/Datasets/Dr(eye)ve/DREYEVE_DATA")
RT_VID_PATH = DS_PATH / REC / "video_garmin.avi"
ETG_VID_PATH = DS_PATH / REC / "video_etg.avi"
ETG_DATA_PATH = DS_PATH / REC / "etg_samples.txt"
GPS_PATH = DS_PATH / REC / "speed_course_coord.txt"
RT_SAMPLE_PATH = pathlib.Path().absolute() / "media" / "rt_camera_sample_5.png"
ETG_SAMPLE_PATH = pathlib.Path().absolute() / "media" / "etg_camera_sample_5.png"

FPS_RT = 25
FPS_ETG = 30

if __name__ == "__main__":
    print(f"{DS_PATH}: {DS_PATH.is_dir()}")
    print(f"{RT_VID_PATH}: {RT_VID_PATH.is_file()}")
    print(f"{ETG_VID_PATH}: {ETG_VID_PATH.is_file()}")
    print(f"{ETG_DATA_PATH}: {ETG_DATA_PATH.is_file()}")
    print(f"{GPS_PATH}: {GPS_PATH.is_file()}")
    print(f"{RT_SAMPLE_PATH}: {RT_SAMPLE_PATH.is_file()}")
    print(f"{ETG_SAMPLE_PATH}: {ETG_SAMPLE_PATH.is_file()}")