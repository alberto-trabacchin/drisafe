import cv2 as cv
from constants import ETG_VID_PATH, RT_VID_PATH, FPS_ETG, FPS_RT, ETG_DATA_PATH

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
        self.etg_cap = cv.VideoCapture(str(self.etg_cam["path"]))
        self.rt_cap = cv.VideoCapture(str(self.rt_cam["path"]))
        self.etg_frame = None
        self.etg_status = None
        self.rt_frame = None
        self.rt_status = None
        self.curr_time = 0
        self.k = 0
        self.online = True

    def sync_frames(self):
        """
        Updates cameras' frames in order to synchronize each other.
        """ 
        rt_cam = self.rt_cam
        etg_cam = self.etg_cam
        (self.etg_status, self.etg_frame) = self.etg_cap.read()
        self.curr_time += 1 / float(etg_cam["fps"])
        if (self.k / float(rt_cam["fps"]) <= self.curr_time):
            self.k += 1
            (self.rt_status, self.rt_frame) = self.rt_cap.read()

    def read(self, show = False):
        """
        Reads cameras' frames and plot them if required.

        Args:
            show: flag to indicate if showing frames is required when reading sensors.
        """ 
        if (self.etg_cap.isOpened() and self.rt_cap.isOpened()):
            self.sync_frames()
            if not (self.etg_status and self.rt_status):
                self.close()
                return
            if show: self.plot_frame()

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



if __name__ == '__main__':
    sensors = {
        "gaze_track": {
            "name": "Gaze Tracker",
            "path": ETG_DATA_PATH
        },
        "etg_cam": {
            "name": "ETG Camera",
            "path": ETG_VID_PATH,
            "fps": FPS_ETG
        },
        "roof_cam": {
            "name": "Roof Top Camera",
            "path": RT_VID_PATH,
            "fps": FPS_RT
        }
    }
    sens_streams = SensorStreams(sensors)

    while True:
        sens_streams.read(show = True)
        if not sens_streams.online: break