import cv2 as cv
import pandas as pd
import numpy as np
import pathlib
from drisafe.constants import SENSORS
from drisafe import homography

class SensorStreams(object):
    """
    SensorStreams class allows to manage all the sensors synchronously.

    Managed sources are from ETG camera, Rooftop camera and ETG tracker.
    ETG camera is recorded at 30fps.
    Rooftop camera is recorded at 25fps.
    ETG tracker tracks gaze at 30Hz.

    Attributes:
        etg_cam: Dictionary which contains all information about ETG camera.
        rt_cam: Dictionary which contains all information about Rooftop camera.
        etg_cap: Capture of ETG camera source
        rt_cap: Capture of Rooftop camera source
        etg_frame: Fps of ETG camera
        etg_status: Reading status of ETG camera (frame available or not)
        rt_frame: Fps of Rooftop camera
        rt_status: Reading status of Rooftop camera (frame available or not)
        k: Coefficient which counts how many times Rooftop camera has been updated.
        online: Flag which indicates if sensors are no longer available for reading.
    """

    def __init__(self, sensor, rec_id, t_step = 0):
        """
        Initializes the instance based on sensors parameters.

        Args:
            sensors: Dictionary which contains sensors' information.
        """
        self.etg_cam = sensor["etg_cam"]
        self.rt_cam = sensor["roof_cam"]
        self.etg_tracker = sensor["gaze_track"]
        self.etg_cap = cv.VideoCapture(str(self.etg_cam["vid_path"]))
        self.rt_cap = cv.VideoCapture(str(self.rt_cam["vid_path"]))
        ds_tracker = pd.read_csv(str(self.etg_tracker["etg_crd_path"]), 
                                          delim_whitespace = True)
        self.ds_tracker = ds_tracker.dropna(axis = "rows", how = "any")
        start_rt_frame = ds_tracker.iloc[t_step]["frame_gar"]
        start_etg_frame = ds_tracker.iloc[t_step]["frame_etg"]
        self.rt_cap.set(cv.CAP_PROP_POS_FRAMES, start_rt_frame)
        self.etg_cap.set(cv.CAP_PROP_POS_FRAMES, start_etg_frame)
        if (self.etg_tracker["rt_crd_path"].is_file()):
            self.ds_rt_crds = pd.read_csv(str(self.etg_tracker["rt_crd_path"]), header = 0)
        else:
            self.ds_rt_crds = pd.DataFrame({"X": [], "Y": []})
        self.etg_crd = np.array([[[0, 0]]])
        self.rt_crd = np.array([[[0, 0]]])
        self.etg_frame = None
        self.etg_status = True
        self.rt_frame = None
        self.rt_status = True
        self.t_step = t_step
        self.online = True
        self.rec_id = rec_id

    def _read_rt_coord_file(self):
        rt_coord_path = self.etg_tracker["rt_coord_paths"][self.rec_id]
        if (rt_coord_path.is_file()):
            ds_rt_tracker = pd.read_csv(str(rt_coord_path), delim_whitespace = True)
            print(self.rt_cap.get(cv.CAP_PROP_FRAME_COUNT))
            if (ds_rt_tracker.shape[0] == self.ds_etg_tracker.shape[0]):
                write_rt_coords = False
                return ds_rt_tracker, write_rt_coords
        write_rt_coords = True
        ds_rt_tracker = pd.DataFrame({"X" : [], "Y" : []})
        return ds_rt_tracker, write_rt_coords

    def sync_data(self):
        """
        Updates cameras' frames and gaze coordinates in order to synchronize them.
        """ 
        if (self.t_step < self.ds_tracker.shape[0]):
            rt_frame_no = self.ds_tracker.iloc[self.t_step]["frame_gar"]
            etg_frame_no = self.ds_tracker.iloc[self.t_step]["frame_etg"]
            self.etg_crd = self.ds_tracker.iloc[self.t_step][["X", "Y"]].to_numpy().reshape(1, 1, 2)
            #self.rt_cap.set(cv.CAP_PROP_POS_FRAMES, rt_frame_no)
            #self.etg_cap.set(cv.CAP_PROP_POS_FRAMES, etg_frame_no)
            if (self.t_step == 0):
                self.rt_status, self.rt_frame = self.rt_cap.read()
                self.etg_status, self.etg_frame = self.etg_cap.read()
            else:
                prev_rt_frame_no = self.ds_tracker.iloc[self.t_step - 1]["frame_gar"]
                prev_etg_frame_no = self.ds_tracker.iloc[self.t_step - 1]["frame_etg"]
                if ((prev_rt_frame_no < rt_frame_no) or (self.rt_crd == np.array([[[0, 0]]])).all()):
                    self.rt_status, self.rt_frame = self.rt_cap.read()
                if ((prev_etg_frame_no < etg_frame_no) or (self.rt_crd == np.array([[[0, 0]]])).all()):
                    self.etg_status, self.etg_frame = self.etg_cap.read()
            if not self.ds_rt_crds.empty:
                self.rt_crd = self.ds_rt_crds.iloc[self.t_step][["X", "Y"]].to_numpy(dtype=np.float32).reshape(1, 1, 2)
            self.t_step += 1
        else:
            self.rt_status = False
            self.etg_status = False

    def read(self, show_gaze_crd = False):
        """
        Reads cameras' frames and plot them if required.

        Args:
            show: flag to indicate if showing frames is required when reading sensors.
        """ 
        if (self.etg_cap.isOpened() and self.rt_cap.isOpened()):
            self.sync_data()
            if not (self.etg_status and self.rt_status):
                self.close()
                return
            if show_gaze_crd: 
                print(f"({self.rec_id}-{self.t_step}) - ETG gaze: {self.etg_crd}")

    def plot_frame(self, show_gazes = False):
        """
        Shows frames on the screen.
        """ 
        etg_frame = np.copy(self.etg_frame)
        rt_frame = np.copy(self.rt_frame)
        if (show_gazes):
            rt_crd = np.copy(self.rt_crd)
            etg_crd = np.copy(self.etg_crd)
            rt_frame = homography.draw_gaze(rt_frame, rt_crd)
            etg_frame = homography.draw_gaze(etg_frame, etg_crd)
        etg_h = etg_frame.shape[0]
        etg_w = etg_frame.shape[1]
        rt_h = int(rt_frame.shape[0])
        rt_w = int(rt_frame.shape[1])
        rt_frame = cv.resize(rt_frame, (rt_w, rt_h))
        etg_frame = cv.resize(etg_frame, (etg_w, etg_h))
        etg_frame_ar = etg_w / float(etg_h)
        etg_h = rt_h
        etg_w = int(etg_frame_ar * rt_h)
        etg_frame = cv.resize(etg_frame, (etg_w, etg_h))
        conc_frames = np.concatenate((etg_frame, rt_frame), axis = 1)
        win_name = f"Cameras - REC {self.rec_id}"
        cv.namedWindow(win_name, cv.WND_PROP_FULLSCREEN)          
        #cv.setWindowProperty(win_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow(win_name, conc_frames)
        print(conc_frames.shape)
        if (cv.waitKey(1) & 0xFF == ord("q")):
            self.close()
        return conc_frames

    def close(self):
        """
        Releases camera captures and set sensors' status in offline mode.
        """ 
        self.etg_cap.release()
        self.rt_cap.release()
        cv.destroyAllWindows()
        self.online = False
        print("Sensors closed.")

if __name__ == '__main__':
    sstream = SensorStreams(SENSORS[10], 11, t_step = 6600)
    while True:
        sstream.read(show_gaze_crd = True)
        sstream.plot_frame(show_gazes = True)
        if not sstream.online: break