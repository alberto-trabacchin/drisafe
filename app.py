from drisafe.constants import SENSORS
from drisafe.sensorstreams import SensorStreams
from drisafe import homography

if __name__ == "__main__":
    sensors = SensorStreams(SENSORS)
    while True:
        sensors.read(show_frames = False, show_gaze = True)
        if not sensors.online: break
        sensors.update_rt_coords(show_hom_info = False, show_proj_gaze = False,
                                 plot_proj_gaze = False, plot_matches = False)