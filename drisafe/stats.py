import numpy as np
from drisafe.constants import SENSORS
from drisafe.sensorstreams import SensorStreams

def analyze_gaze_areas(sensors, nx = 4, ny = 3, verbose = False):
    rt_coords = sensors.rt_coords
    rt_frame = sensors.rt_frame
    (h, w, _) = rt_frame.shape
    x_lims = np.arange(start = 0, step = int(w / nx), stop = w)
    y_lims = np.arange(start = 0, step = int(h / ny), stop = h)
    coord = rt_coords[0]
    x_count = np.count_nonzero(coord[0] >= x_lims)
    y_count = np.count_nonzero(coord[1] >= y_lims)
    area_id = nx * (y_count - 1) + x_count
    if verbose:
        print(f"{coord} -> {area_id} / {nx * ny}")
    return area_id

if __name__ == "__main__":
    sensors = SensorStreams(SENSORS)
    sensors.read()
    sensors.update_rt_coords()
    analyze_gaze_areas(sensors, verbose = True)
    sensors.read()
    sensors.update_rt_coords()
    analyze_gaze_areas(sensors, verbose = True)