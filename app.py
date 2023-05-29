from drisafe.constants import SENSORS
from drisafe.sensorstreams import SensorStreams
from drisafe import homography

if __name__ == "__main__":
    sensors = SensorStreams(SENSORS)
    while True:
        sensors.read(show_frames = True, show_gaze = True)
        if not sensors.online: break
        sensors.calc_rt_coords(verbose = True)