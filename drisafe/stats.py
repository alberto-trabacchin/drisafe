import numpy as np
from drisafe.constants import SENSORS
from drisafe.sensorstreams import SensorStreams

class Stats(object):
    def __init__(self, nx = 4, ny = 3):
        self.nx = nx
        self.ny = ny
        self.gaze_areas_count = np.zeros((nx * ny), dtype = np.int32)

    def find_gaze_area(self, sensors, verbose = False):
        rt_coords = sensors.rt_coords
        rt_frame = sensors.rt_frame
        (h, w, _) = rt_frame.shape
        x_lims = np.arange(start = 0, step = int(w / self.nx), stop = w)
        y_lims = np.arange(start = 0, step = int(h / self.ny), stop = h)
        coord = rt_coords[0]

        x_count = np.count_nonzero(coord[0] >= x_lims)
        y_count = np.count_nonzero(coord[1] >= y_lims)
        area_id = self.nx * (y_count - 1) + x_count - 1
        if verbose:
            print(f"{coord} -> {area_id} / {self.nx * self.ny - 1}")
            print(f"{self.gaze_areas_count}")
        self.gaze_areas_count[area_id] += 1
        return area_id

if __name__ == "__main__":
    sensors = SensorStreams(SENSORS)
    stats = Stats(nx = 3, ny = 3)
    sensors.read()
    sensors.update_rt_coords()
    stats.find_gaze_area(sensors, verbose = True)
    sensors.read()
    sensors.update_rt_coords()
    stats.find_gaze_area(sensors, verbose = True)