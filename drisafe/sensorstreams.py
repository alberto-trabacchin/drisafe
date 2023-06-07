import cv2 as cv
import pandas as pd
import numpy as np
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

    def __init__(self, sensors, start_id = 1):
        """
        Initializes the instance based on sensors parameters.

        Args:
            sensors: Dictionary which contains sensors' information.
        """
        self.etg_cam = sensors["etg_cam"]
        self.rt_cam = sensors["roof_cam"]
        self.etg_tracker = sensors["gaze_track"]
        self.rec_id = start_id - 1
        self.etg_cap = cv.VideoCapture(str(self.etg_cam["paths"][self.rec_id]))
        self.rt_cap = cv.VideoCapture(str(self.rt_cam["paths"][self.rec_id]))
        self.ds_etg_tracker = pd.read_csv(str(self.etg_tracker["paths"][self.rec_id]), 
                                              delim_whitespace = True)
        self.ds_etg_tracker = self.ds_etg_tracker[["X", "Y"]]
        self.rec_tot_frames = self.etg_cap.get(cv.CAP_PROP_FRAME_COUNT)
        self.etg_coords = None
        self.rt_coords = None
        self.etg_frame = None
        self.etg_status = None
        self.rt_frame = None
        self.rt_status = None
        self.t_step = 0
        self.k = 0
        self.online = True

    def _read_next_rec(self):
        self.rec_id += 1
        self.etg_cap.close()
        self.rt_cap.close()
        self.etg_cap = cv.VideoCapture(str(self.etg_cam["paths"][self.rec_id]))
        self.rt_cap = cv.VideoCapture(str(self.rt_cam["paths"][self.rec_id]))
        self.ds_etg_tracker = pd.read_csv(str(self.etg_tracker["paths"][self.rec_id]), 
                                              delim_whitespace = True)

    def sync_data(self):
        """
        Updates cameras' frames and gaze coordinates in order to synchronize them.
        """ 
        rt_cam = self.rt_cam
        etg_cam = self.etg_cam
        ds_etg_tracker = self.ds_etg_tracker
        t_step = self.t_step
        self.etg_coords = ds_etg_tracker.iloc[t_step].to_numpy().reshape(-1, 2)
        (self.etg_status, self.etg_frame) = self.etg_cap.read()
        self.t_step += 1
        t_rt = self.k / float(rt_cam["fps"])
        t_ref = self.t_step / float(etg_cam["fps"])
        if (t_rt <= t_ref):
            self.k += 1
            (self.rt_status, self.rt_frame) = self.rt_cap.read()

    def get_recs_len(self):
        return len(self.rt_cam["paths"])
    
    def _check_rec_finished(self):
        if (self.t_step == self.rec_tot_frames):
            if (self.rec_id < self.get_recs_len() - 1):
                self._open_next_rec()
            else:
                print("All recordings have been red.")

    def _open_next_rec(self, show_frames = False, show_gaze = False):
        self.rec_id += 1
        self.t_step = 0
        self.k = 0
        self.etg_cap = cv.VideoCapture(str(self.etg_cam["paths"][self.rec_id]))
        self.rt_cap = cv.VideoCapture(str(self.rt_cam["paths"][self.rec_id]))
        self.ds_etg_tracker = pd.read_csv(str(self.etg_tracker["paths"][self.rec_id]), 
                                                delim_whitespace = True)
        self.ds_etg_tracker = self.ds_etg_tracker[["X", "Y"]]
        self.rec_tot_frames = self.etg_cap.get(cv.CAP_PROP_FRAME_COUNT)

    def read(self, show_frames = False, show_gaze = False):
        """
        Reads cameras' frames and plot them if required.

        Args:
            show: flag to indicate if showing frames is required when reading sensors.
        """ 
        if (self.etg_cap.isOpened() and self.rt_cap.isOpened()):
            self._check_rec_finished()            
            self.sync_data()
            if not (self.etg_status and self.rt_status):
                self.close()
                return
            if show_gaze: print(f"ETG gaze: {self.etg_coords}")
            if show_frames: self.plot_frame()

    def plot_frame(self):
        """
        Shows frames on the screen.
        """ 
        etg_frame = self.etg_frame
        rt_frame = self.rt_frame
        etg_h = etg_frame.shape[0]
        etg_w = etg_frame.shape[1]
        rt_h = int(0.7 * rt_frame.shape[0])
        rt_w = int(0.7 * rt_frame.shape[1])
        rt_frame = cv.resize(rt_frame, (rt_w, rt_h))
        etg_frame = cv.resize(etg_frame, (etg_w, etg_h))
        etg_frame_ar = etg_w / float(etg_h)
        etg_h = rt_h
        etg_w = int(etg_frame_ar * rt_h)
        etg_frame = cv.resize(etg_frame, (etg_w, etg_h))
        conc_frames = np.concatenate((etg_frame, rt_frame), axis = 1)
        cv.imshow("Cameras", conc_frames)
        if (cv.waitKey(1) & 0xFF == ord("q")):
            self.close()

    def close(self):
        """
        Releases camera captures and set sensors' status in offline mode.
        """ 
        self.etg_cap.release()
        self.rt_cap.release()
        cv.destroyAllWindows()
        self.online = False
        print("Sensors closed.")

    def update_rt_coords(self, show_hom_info = False, show_proj_gaze = False, 
                         plot_proj_gaze = False, plot_matches = False):
        rt_frame_gray = cv.cvtColor(self.rt_frame, cv.COLOR_BGR2GRAY)
        etg_frame_gray = cv.cvtColor(self.etg_frame, cv.COLOR_BGR2GRAY)
        rt_kp, rt_des = homography.SIFT(rt_frame_gray)
        etg_kp, etg_des = homography.SIFT(etg_frame_gray)
        matches = homography.match_keypoints(etg_des, rt_des, threshold = homography.KNN_THRESH)
        H, mask = homography.estimate_homography(etg_kp, rt_kp, matches, show_hom_info)
        self.rt_coords = homography.project_gaze(self.etg_coords, H)
        if (show_proj_gaze):
            print(f"RT gaze: {self.rt_coords}")
        if (plot_proj_gaze):
           ret = homography.print_gaze(self.rt_frame, self.etg_frame, self.rt_coords, self.etg_coords)
           if (ret == False): self.close()
        if (plot_matches):
            homography. plot_matches(matches, etg_kp, rt_kp, self.etg_frame, self.rt_frame, mask)



if __name__ == '__main__':
    sens_streams = SensorStreams(SENSORS, start_id = 74)
    while True:
        sens_streams.read(show_frames = True, show_gaze = True)
        if not sens_streams.online: break