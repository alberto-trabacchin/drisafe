import cv2 as cv
import pandas as pd
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

    def __init__(self, sensors):
        """
        Initializes the instance based on sensors parameters.

        Args:
            sensors: Dictionary which contains sensors' information.
        """
        self.etg_cam = sensors["etg_cam"]
        self.rt_cam = sensors["roof_cam"]
        self.etg_tracker = sensors["gaze_track"]
        self.etg_cap = cv.VideoCapture(str(self.etg_cam["path"]))
        self.rt_cap = cv.VideoCapture(str(self.rt_cam["path"]))
        self.ds_etg_tracker = pd.read_csv(str(self.etg_tracker["path"]), delim_whitespace = True)
        self.ds_etg_tracker = self.ds_etg_tracker[["X", "Y"]]
        self.etg_coords = None
        self.rt_coords = None
        self.etg_frame = None
        self.etg_status = None
        self.rt_frame = None
        self.rt_status = None
        self.t_step = 0
        self.k = 0
        self.online = True

    def sync_data(self):
        """
        Updates cameras' frames and gaze coordinates in order to synchronize them.
        """ 
        rt_cam = self.rt_cam
        etg_cam = self.etg_cam
        ds_etg_tracker = self.ds_etg_tracker
        t_step = self.t_step
        self.etg_coords = ds_etg_tracker.iloc[t_step].to_numpy()
        (self.etg_status, self.etg_frame) = self.etg_cap.read()
        self.t_step += 1
        t_rt = self.k / float(rt_cam["fps"])
        t_ref = self.t_step / float(etg_cam["fps"])
        if (t_rt <= t_ref):
            self.k += 1
            (self.rt_status, self.rt_frame) = self.rt_cap.read()

    def read(self, show_frames = False, show_gaze = False):
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
            if show_gaze: print(f"ETG gaze: {self.etg_coords}")
            if show_frames: self.plot_frame()

    def plot_frame(self):
        """
        Shows frames on the screen.
        """ 
        cv.imshow(self.etg_cam["name"], self.etg_frame)
        cv.imshow(self.rt_cam["name"], self.rt_frame)
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
        self.rt_coords = homography.project_gaze(self.etg_coords, H).reshape(2)
        if (show_proj_gaze):
            print(f"RT gaze: {self.rt_coords}")
        if (plot_proj_gaze):
           homography.print_gaze(self.etg_frame, self.rt_frame, self.etg_coords, self.rt_coords)
        if (plot_matches):
            homography. plot_matches(matches, etg_kp, rt_kp, self.etg_frame, self.rt_frame, mask)



if __name__ == '__main__':
    sens_streams = SensorStreams(SENSORS)
    while True:
        sens_streams.read(show_frames = True, show_gaze = True)
        if not sens_streams.online: break