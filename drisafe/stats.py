import numpy as np
from drisafe.constants import SENSORS
from drisafe.sensorstreams import SensorStreams
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt

class Stats(object):
    def __init__(self, nx = 4, ny = 3):
        self.nx = nx
        self.ny = ny
        self.gaze_areas_count = np.zeros((ny + 2, nx + 2), dtype = np.int32)

    def calc_gaze_matrix(self, sstream, verbose = False):
        rt_coords = sstream.rt_crd
        rt_frame = sstream.rt_frame
        (h, w, _) = rt_frame.shape
        x_lims = np.linspace(start = 0, stop = w, num = self.nx + 1)
        y_lims = np.linspace(start = 0, stop = h, num = self.ny + 1)
        coord = rt_coords[0][0]
        x_count = np.count_nonzero(coord[0] >= x_lims)
        y_count = np.count_nonzero(coord[1] >= y_lims)
        self.gaze_areas_count[y_count, x_count] += 1
        if verbose:
            print(f"{self.gaze_areas_count}")

    def calc_all_gaze_matrix(self, sstream, verbose = False):
        rt_frame = sstream.rt_frame
        (h, w, _) = rt_frame.shape
        x_lims = np.linspace(start = 0, stop = w, num = self.nx + 1)
        y_lims = np.linspace(start = 0, stop = h, num = self.ny + 1)
        for idx, rt_crd in sstream.ds_rt_crds.iterrows():
            rt_crd = rt_crd[["X", "Y"]].to_numpy(dtype = np.float32)
            x_count = np.count_nonzero(rt_crd[0] >= x_lims)
            y_count = np.count_nonzero(rt_crd[1] >= y_lims)
            self.gaze_areas_count[y_count, x_count] += 1
            if verbose:
                print(f"({sstream.rec_id}-{idx}) --> {x_count, y_count}")
        return self
    
def worker(id, nx, ny, ret_dic):
    sstream = SensorStreams(SENSORS[id], id)
    stats = Stats(nx, ny)
    sstream.read()
    stats.calc_all_gaze_matrix(sstream, verbose = False)
    ret_dic[id] = {
        "rec_id": id + 1,
        "gaze_mat": stats.gaze_areas_count
        }
    
def plot_data(gaze_mat):
    print(f"Plotting recording No. {rec_id}.")
    sns.heatmap(gaze_mat, vmin = 0, vmax = 5)
    

if __name__ == "__main__":
    nx, ny = 700, 300
    rec_ids = range(10, 15)
    manager = mp.Manager()
    ret_dic = manager.dict()
    jobs = []
    for id in rec_ids:
        p = mp.Process(target = worker, args = (id, nx, ny, ret_dic))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    for res in ret_dic.values():
        rec_id = res["rec_id"]
        gaze_mat = res["gaze_mat"]
        print(f"Recording ID: {rec_id}")
        print(f"Total samples: {np.sum(gaze_mat)}")
        print(f"{gaze_mat} \n")
        sns.heatmap(gaze_mat, vmin = 0, vmax = 5)
        plt.show()
        print(f"{rec_id}")