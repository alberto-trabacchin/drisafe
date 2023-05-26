from threading import Thread
import cv2 as cv
import time
from constants import ETG_VID_PATH, RT_VID_PATH, FPS_ETG, FPS_RT

class CamStreams(object):
    def __init__(self, cameras):
        self.etg_cam = cameras["etg"]
        self.rt_cam = cameras["roof"]
        self.etg_cap = cv.VideoCapture(str(self.etg_cam["path"]))
        self.rt_cap = cv.VideoCapture(str(self.rt_cam["path"]))
        self.curr_time = 0
        self.k = 0
        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()

    def first_read(self):
        if not (self.etg_cap.isOpened() and self.rt_cap.isOpened()):
            raise SystemExit("Error opening cameras.")
        (self.etg_status, self.etg_frame) = self.etg_cap.read()
        (self.rt_status, self.rt_frame) = self.rt_cap.read()

    def read_sync_frames(self):
        rt_cam = self.rt_cam
        etg_cam = self.etg_cam
        (self.etg_status, self.etg_frame) = self.etg_cap.read()
        self.curr_time += 1 / float(etg_cam["fps"])
        if (self.k / float(rt_cam["fps"]) < self.curr_time):
            self.k += 1
            (self.rt_status, self.rt_frame) = self.rt_cap.read()

    def update(self):
        self.first_read()
        while (self.etg_cap.isOpened() and self.rt_cap.isOpened()):
            self.read_sync_frames()
            time.sleep(.01)

    def close(self):
        self.etg_cap.release()
        self.rt_cap.release()
        cv.destroyAllWindows()
        exit(1)

    def show_frame(self):
        cv.imshow(self.etg_cam["name"], self.etg_frame)
        cv.imshow(self.rt_cam["name"], self.rt_frame)
        if (cv.waitKey(1) & 0xFF == ord("q")):
            self.close()


if __name__ == '__main__':
    cameras = {
        "etg": {
            "name": "ETG camera",
            "path": ETG_VID_PATH,
            "fps": FPS_ETG
        },
        "roof": {
            "name": "Roof top camera",
            "path": RT_VID_PATH,
            "fps": FPS_RT
        }
    }
    camera_streams = CamStreams(cameras)
    while True:
        try:
            camera_streams.show_frame()
        except AttributeError:
            pass