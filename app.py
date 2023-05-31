from drisafe.constants import SENSORS
from drisafe.sensorstreams import SensorStreams
from drisafe.stats import Stats

if __name__ == "__main__":
    sensors = SensorStreams(SENSORS)
    stats = Stats(nx = 7, ny = 3)
    while True:
        sensors.read(show_frames = False, show_gaze = False)
        if not sensors.online: break
        sensors.update_rt_coords(show_hom_info = False, show_proj_gaze = False,
                                 plot_proj_gaze = True, plot_matches = False)
        stats.find_gaze_area(sensors, verbose = True)
        