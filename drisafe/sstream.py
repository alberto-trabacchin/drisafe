import cv2 as cv
import pandas as pd
import numpy as np
from drisafe.constants import SENSORS
from drisafe import homography
import json

class Camera(object):

    def __init__(self, rec_id, model):
        self.rec_id = rec_id
        sensor = SENSORS[rec_id - 1]
        cam = sensor[model]
        if (model == "roof_cam"):
            self.model = "RT cam"
        else:
            self.model = "ETG cam"
        self.cap = cv.VideoCapture(str(cam["vid_path"]))
        w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame = np.zeros(shape = (h, w, 3), dtype = np.int8)
        self.gaze_crd = np.zeros(shape = (1, 1, 2), dtype = np.float16)
        self.status = True

    def set_start_frame(self, frame_no):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_no - 1)
        print(f"{self.model} set to frame {frame_no}.")

    def get_frame_count(self):
        return self.cap.get(cv.CAP_PROP_POS_FRAMES)
    
    def get_camera_params(self):
        fps = self.cap.get(cv.CAP_PROP_FPS)
        width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        return fps, width, height

    def read_frame(self):
        self.status, self.frame = self.cap.read()
    
    def close(self):
        self.cap.release()
        cv.destroyAllWindows()
        print(f"({self.rec_id}) {self.model} closed.")
    

class SStream(object):
    
    def __init__(self, rec_id, t_step = 0):
        self.rec_id = rec_id
        sensor = SENSORS[rec_id - 1]
        self.rt_cam = Camera(rec_id, model = "roof_cam")
        self.etg_cam = Camera(rec_id, model = "etg_cam")
        etg_data_path = sensor["gaze_track"]["etg_crd_path"]
        rt_data_path = sensor["gaze_track"]["rt_crd_path"]
        self.etg_data = pd.read_csv(etg_data_path, 
                                    delim_whitespace = True,
                                    index_col = False)
        if (rt_data_path.is_file()):
            f = open(rt_data_path)
            self.rt_data = json.load(f)
            self.rt_data_avail = True
            f.close()
        else:
            self.rt_data = pd.DataFrame({"X": [], "Y": []})
            self.rt_data_avail = False
        self.t_step = t_step
        self.online = True

    def sync_data(self):
        if (self.t_step < self.etg_data.shape[0]):
            data = self.etg_data.iloc[self.t_step]
            etg_frame_id = data["frame_etg"]
            rt_frame_id = data["frame_gar"]
            self.etg_cam.gaze_crd = np.array([
                    data["X"], data["Y"]
                ]).astype(np.float32).reshape(1, 1, 2)
            while (self.rt_cam.get_frame_count() <= rt_frame_id):
                self.rt_cam.read_frame()
            while (self.etg_cam.get_frame_count() <= etg_frame_id):
                self.etg_cam.read_frame()
            if (self.rt_data_avail) and (self.t_step < len(self.rt_data)):
                self.rt_cam.gaze_crd = np.array(
                    self.rt_data[self.t_step]["rt_crd"]
                ).reshape(1, 1, 2)
            self.t_step += 1
        else:
            self.online = False

    def read(self):
        if (self.rt_cam.cap.isOpened() and self.etg_cam.cap.isOpened()):
            self.sync_data()
        else:
            self.online = False

    def show_coordinates(self):
        print(f"({self.rec_id}, {self.t_step}) - ETG coords: {self.etg_cam.gaze_crd}")
        print(f"({self.rec_id}, {self.t_step}) - RT coords: {self.rt_cam.gaze_crd}")

    def valid_coords(self):
        return not (np.isnan(np.sum(self.rt_cam.gaze_crd)) or np.isnan(np.sum(self.etg_cam.gaze_crd)))

    def show_frames_side(self, fullscreen = False, show_gazes = False):
        etg_frame = np.copy(self.etg_cam.frame)
        rt_frame = np.copy(self.rt_cam.frame)
        if (show_gazes and self.valid_coords()):
            rt_crd = np.copy(self.rt_cam.gaze_crd)
            etg_crd = np.copy(self.etg_cam.gaze_crd)
            rt_frame = homography.draw_gaze(rt_frame, rt_crd)
            etg_frame = homography.draw_gaze(etg_frame, etg_crd)
        etg_h = etg_frame.shape[0]
        etg_w = etg_frame.shape[1]
        if fullscreen:
            rt_h = int(rt_frame.shape[0])
            rt_w = int(rt_frame.shape[1])
        else:
            rt_h = int(0.5 * rt_frame.shape[0])
            rt_w = int(0.5 * rt_frame.shape[1])
        rt_frame = cv.resize(rt_frame, (rt_w, rt_h))
        etg_frame = cv.resize(etg_frame, (etg_w, etg_h))
        etg_frame_ar = etg_w / float(etg_h)
        etg_h = rt_h
        etg_w = int(etg_frame_ar * rt_h)
        etg_frame = cv.resize(etg_frame, (etg_w, etg_h))
        conc_frames = np.concatenate((etg_frame, rt_frame), axis = 1)
        win_name = f"REC{self.rec_id}"
        if fullscreen:
            cv.namedWindow(win_name, cv.WND_PROP_FULLSCREEN)
            cv.setWindowProperty(win_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow(win_name, conc_frames)
        if (cv.waitKey(1) & 0xFF == ord("q")):
            self.online = False
    
    def show_frames(self):
        cv.imshow("RT CAM", self.rt_cam.frame)
        cv.imshow("ETG CAM", self.etg_cam.frame)
        if (cv.waitKey(1) & 0xFF == ord("q")):
            self.online = False

    def set_init_frames(self, rt_frame_no):
        data = self.etg_data
        sel_data = data.loc[data["frame_gar"] >= rt_frame_no]
        sample = sel_data.iloc[0]
        self.t_step = sample.name
        etg_frame_no = sample.frame_etg
        rt_frame_no = sample.frame_gar
        self.rt_cam.set_start_frame(rt_frame_no)
        self.etg_cam.set_start_frame(etg_frame_no)

    def close(self):
        self.rt_cam.close()
        self.etg_cam.close()
        


if __name__ == "__main__":
    stream = SStream(rec_id = 6)
    stream.set_init_frames(rt_frame_no = 0)
    while stream.online:
        stream.read()
        stream.show_frames_side(fullscreen = True,
                                show_gazes = True)
        stream.show_coordinates()
    stream.close()

