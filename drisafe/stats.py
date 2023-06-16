import numpy as np
from drisafe.constants import SENSORS
from drisafe.sensorstreams import SensorStreams

class Stats(object):
    def __init__(self, nx = 4, ny = 3):
        self.nx = nx
        self.ny = ny
        self.gaze_areas_count = np.zeros((ny + 2, nx + 2), dtype = np.int32)

    def find_gaze_area(self, sensor, verbose = False):
        rt_coords = sensor.rt_crd
        rt_frame = sensor.rt_frame
        (h, w, _) = rt_frame.shape
        x_lims = np.linspace(start = 0, stop = w, num = self.nx + 1)
        y_lims = np.linspace(start = 0, stop = h, num = self.ny + 1)
        coord = rt_coords[0][0]
        x_count = np.count_nonzero(coord[0] >= x_lims)
        y_count = np.count_nonzero(coord[1] >= y_lims)
        self.gaze_areas_count[y_count, x_count] += 1
        if verbose:
            print(f"{self.gaze_areas_count}")

if __name__ == "__main__":
    sensors = SensorStreams(SENSORS[0])
    stats = Stats(nx = 3, ny = 3)
    sensors.read()
    sensors.update_rt_coords()
    stats.find_gaze_area(sensors, verbose = True)
    sensors.read()
    sensors.update_rt_coords()
    stats.find_gaze_area(sensors, verbose = True)