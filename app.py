from drisafe.constants import SENSORS
from drisafe.sensorstreams import SensorStreams

if __name__ == "__main__":
    sensors = SensorStreams(SENSORS)
    while True:
        sensors.read(show = True)
        if not sensors.online: break