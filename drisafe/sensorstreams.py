import cv2 as cv
from constants import ETG_VID_PATH, RT_VID_PATH, FPS_ETG, FPS_RT, ETG_DATA_PATH

class SensorStreams(object):
    def __init__(self, sensors):
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
        rt_cam = self.rt_cam
        etg_cam = self.etg_cam
        (self.etg_status, self.etg_frame) = self.etg_cap.read()
        self.curr_time += 1 / float(etg_cam["fps"])
        if (self.k / float(rt_cam["fps"]) <= self.curr_time):
            self.k += 1
            (self.rt_status, self.rt_frame) = self.rt_cap.read()

    def read(self, show = False):
        if (self.etg_cap.isOpened() and self.rt_cap.isOpened()):
            self.sync_frames()
            if show: self.plot_frame()

    def plot_frame(self):
        cv.imshow(self.etg_cam["name"], self.etg_frame)
        cv.imshow(self.rt_cam["name"], self.rt_frame)
        if (cv.waitKey(1) & 0xFF == ord("q")):
            self.close()

    def close(self):
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