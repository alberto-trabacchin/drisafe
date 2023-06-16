import unittest
import numpy as np
from drisafe import stats
from drisafe.constants import SENSORS
from drisafe.sensorstreams import SensorStreams
from drisafe.compute_rt_crds import calc_rt_crds

class TestStats(unittest.TestCase):
    def test_constructor(self):
        nx, ny = 3, 4
        stat = stats.Stats(nx, ny)
        self.assertEqual(stat.nx, nx, f"Should be {nx}.")
        self.assertEqual(stat.ny, ny, f"Should be {ny}.")
        np.testing.assert_array_equal(stat.gaze_areas_count, np.zeros((ny + 2, nx + 2)),
                                      "Wrong areas array init.")
        self.assertEqual(stat.gaze_areas_count.dtype, np.int32)

    def test_find_gaze_area(self):
        stat = stats.Stats(nx = 3, ny = 4)
        sensor = SensorStreams(SENSORS[0], 0)
        sensor.read()
        rt_crds = calc_rt_crds(sensor)

        sensor.rt_crd = rt_crds #Temporary
        ref_gaze_areas_count = np.zeros((stat.ny + 2, stat.nx + 2))
        ref_gaze_areas_count[2, 2] = 1
        stat.find_gaze_area(sensor, verbose = False)
        np.testing.assert_array_equal(stat.gaze_areas_count, ref_gaze_areas_count, 
                                      "Area ID should be 4.")

        sensor.rt_crd = np.array([[[0, 0]]])
        ref_gaze_areas_count.fill(0)
        stat.gaze_areas_count.fill(0)
        ref_gaze_areas_count[1, 1] = 1
        stat.find_gaze_area(sensor, verbose = False)
        np.testing.assert_array_equal(stat.gaze_areas_count, ref_gaze_areas_count, 
                                      "Not matching case (0, 0).")

        sensor.rt_crd = np.array([[[-10, 0]]])
        ref_gaze_areas_count.fill(0)
        stat.gaze_areas_count.fill(0)
        ref_gaze_areas_count[1, 0] = 1
        stat.find_gaze_area(sensor, verbose = False)
        np.testing.assert_array_equal(stat.gaze_areas_count, ref_gaze_areas_count, 
                                      "Not matching case (-10, 0).")

        sensor.rt_crd = np.array([[[-10, -10]]])
        ref_gaze_areas_count.fill(0)
        stat.gaze_areas_count.fill(0)
        ref_gaze_areas_count[0, 0] = 1
        stat.find_gaze_area(sensor, verbose = False)
        np.testing.assert_array_equal(stat.gaze_areas_count, ref_gaze_areas_count, 
                                      "Not matching case (-10, -10).")
        
        sensor.rt_crd = np.array([[[2000, 2000]]])
        ref_gaze_areas_count.fill(0)
        stat.gaze_areas_count.fill(0)
        ref_gaze_areas_count[5, 4] = 1
        stat.find_gaze_area(sensor, verbose = False)
        np.testing.assert_array_equal(stat.gaze_areas_count, ref_gaze_areas_count, 
                                      "Not matching case (2000, 2000).")
        
        sensor.rt_crd = np.array([[[640, 810]]])
        ref_gaze_areas_count.fill(0)
        stat.gaze_areas_count.fill(0)
        ref_gaze_areas_count[4, 2] = 1
        stat.find_gaze_area(sensor, verbose = False)
        np.testing.assert_array_equal(stat.gaze_areas_count, ref_gaze_areas_count, 
                                      "Not matching case (2000, 2000).")
        

if __name__ == "__main__":
    unittest.main()