import numpy as np
from drisafe.constants import SENSORS
from drisafe.config.paths import DS_DESIGN_PATH, RESULTS_PATH
from drisafe.sensorstreams import SensorStreams
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

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
    
def show_gaze_areas(gaze_mat):
    plt.imshow(gaze_mat)
    plt.show()
    
def worker(id, nx, ny, ret_dic):
    sstream = SensorStreams(SENSORS[id], id)
    stats = Stats(nx, ny)
    sstream.read()
    stats.calc_all_gaze_matrix(sstream, verbose = False)
    ret_dic[id] = {
        "rec_id": id + 1,
        "gaze_mat": stats.gaze_areas_count
        }

def read_ds_design():
    col_names = ["rec_id", "time", "weather", "area", "driver_id", "test"]
    df_design = pd.read_csv(DS_DESIGN_PATH, names = col_names, header = None, 
                            sep='\t')
    return df_design

def plot_area_driver_distrib(df):
    fontsize = 12
    ticksize = 10
    labelsize = 10
    plt.clf()
    rcParams['figure.figsize'] = 2, 2
    p = sns.countplot(data=df, x='driver_id', hue='area', edgecolor=None,
                      palette=['#CD2626',"green", "darkblue"],
                      order = ["D1", "D2", "D3", "D4", "D5", "D6"],
                      hue_order=["Highway", "Countryside", "Downtown"])
    p.legend(title='Area', bbox_to_anchor=(1, 1), loc='upper left')
    plt.setp(p.get_legend().get_texts(), fontsize=labelsize) # for legend text
    plt.setp(p.get_legend().get_title(), fontsize=labelsize) # for legend title
    p.set_xticklabels(["D1", "D2", "D3", "D4", "D5", "D6"], size = ticksize)
    for c in p.containers:
        # set the bar label
        p.bar_label(c, fmt='%.0f', label_type='edge')
    plt.yticks([])
    plt.xlabel("Driver ID", fontsize = fontsize)
    plt.ylabel("# Recordings", fontsize = fontsize)
    plt.tight_layout()
    plt.show()
    #plt.savefig(RESULTS_PATH / "area_driver_distrib.png", dpi = 800)

def plot_daytime_driver_distrib(df):
    fontsize = 12
    ticksize = 10
    labelsize = 10
    plt.clf()
    rcParams['figure.figsize'] = 2, 2
    p = sns.countplot(data=df, x='driver_id', hue='time', edgecolor=None,
                      palette=['#CD2626',"green", "darkblue"],
                      order = ["D1", "D2", "D3", "D4", "D5", "D6"],
                      hue_order=["Morning", "Evening", "Night"])
    p.legend(title='Daytime', bbox_to_anchor=(1, 1), loc='upper left')
    plt.setp(p.get_legend().get_texts(), fontsize=labelsize) # for legend text
    plt.setp(p.get_legend().get_title(), fontsize=labelsize) # for legend title
    p.set_xticklabels(["D1", "D2", "D3", "D4", "D5", "D6"], size = ticksize)
    for c in p.containers:
        p.bar_label(c, fmt='%.0f', label_type='edge')
    plt.yticks([])
    plt.xlabel("Driver ID", fontsize = fontsize)
    plt.ylabel("# Recordings", fontsize = fontsize)
    plt.tight_layout()
    plt.show()
    #plt.savefig(RESULTS_PATH / "daytime_driver_distrib.png", dpi = 800)

def plot_weather_driver_distrib(df):
    fontsize = 12
    ticksize = 10
    labelsize = 10
    plt.clf()
    rcParams['figure.figsize'] = 2, 2
    p = sns.countplot(data=df, x='driver_id', hue='weather', edgecolor=None,
                      palette=['#CD2626',"green", "darkblue"],
                      order = ["D1", "D2", "D3", "D4", "D5", "D6"],
                      hue_order=["Sunny", "Cloudy", "Rainy"])
    p.legend(title='Weather', bbox_to_anchor=(1, 1), loc='upper left')
    plt.setp(p.get_legend().get_texts(), fontsize=labelsize) # for legend text
    plt.setp(p.get_legend().get_title(), fontsize=labelsize) # for legend title
    p.set_xticklabels(["D1", "D2", "D3", "D4", "D5", "D6"], size = ticksize)
    for c in p.containers:
        p.bar_label(c, fmt='%.0f', label_type='edge')
    plt.yticks([])
    plt.xlabel("Driver ID", fontsize = fontsize)
    plt.ylabel("# Recordings", fontsize = fontsize)
    plt.tight_layout()
    plt.show()
    #plt.savefig(RESULTS_PATH / "weather_driver_distrib.png", dpi = 800)

def plot_data(gaze_mat):
    print(f"Plotting recording No. {rec_id}.")
    sns.heatmap(gaze_mat, vmin = 0, vmax = 1)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    

if __name__ == "__main__":
    df_design = read_ds_design()
    #plot_area_driver_distrib(df_design)
    #plot_daytime_driver_distrib(df_design)
    #plot_weather_driver_distrib(df_design)
    nx, ny = 700, 300
    rec_ids = range(0, 74)
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
    sor_ret = sorted(ret_dic.values(), key = lambda x : x["rec_id"])
    plot_data(sor_ret[0]["gaze_mat"])
