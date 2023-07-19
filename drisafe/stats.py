import numpy as np
from drisafe.constants import SENSORS
from drisafe.config.paths import DS_DESIGN_PATH, TRACKING_DATA_PATH
from drisafe.sensorstreams import SensorStreams
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from matplotlib import rcParams
import json
from collections import Counter

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

def plot_gaze(sample):
    gaze_mat = sample["gaze_mat"]
    weather = sample["weather"]
    time = sample["time"]
    area = sample["area"]
    sns.heatmap(gaze_mat, vmin = 0, vmax = 5, cbar = True)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"{area}, {time}, {weather}")
    plt.show()

def select_subset_gazes(gaze_list, df, weather, time, area, nx, ny):
    ids = df.loc[(df["weather"] == weather) & (df["time"] == time) & (df["area"] == area)].index
    subset_gazes = [gaze_list[i]["gaze_mat"] for i in ids if i < len(gaze_list)]
    if not subset_gazes:
        gaze_mat = np.zeros(shape = (ny, nx))
    else:
        gaze_mat = sum(subset_gazes)
    return dict(
        gaze_mat = gaze_mat,
        weather = weather,
        time = time,
        area = area
    )

def plot_gaze_groups(gaze_list, df_design, nx, ny):
    gaze_groups_list = []
    weathers = df_design["weather"].unique()
    times = df_design["time"].unique()
    areas = df_design["area"].unique()
    for scenario in list(itertools.product(weathers, times, areas)):
        w, t, a = scenario
        gaze_sub_dic = select_subset_gazes(gaze_list, df_design, w, t, a, nx, ny)
        gaze_groups_list.append(gaze_sub_dic)
    for gg in gaze_groups_list:
        if gg["gaze_mat"].any() != 0:
            plot_gaze(gg)

def compute_gaze_areas_distribution(df_design):
    plot_area_driver_distrib(df_design)
    plot_daytime_driver_distrib(df_design)
    plot_weather_driver_distrib(df_design)

def compute_gaze_matrices(nx, ny):
    rec_ids = range(0, 75)
    manager = mp.Manager()
    ret_dic = manager.dict()
    jobs = []
    for id in rec_ids:
        p = mp.Process(target = worker, args = (id, nx, ny, ret_dic))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    gaze_list = sorted(ret_dic.values(), key = lambda x : x["rec_id"])
    return gaze_list

def read_tracking_data(rec_ids):    
    track_data = []
    for id in rec_ids:
        data_path = TRACKING_DATA_PATH[id - 1]
        with open(data_path) as json_file:
            data = json.load(json_file)
            track_data.append(data)
    return track_data

def get_obsv_data(track_data):
    obsv_data = []
    for rec_data in track_data:
        for person in rec_data:
            obsv_data.append(person["observed"])
    return obsv_data

def count_obsv_data(obsv_data):
    pos_obsv_counter = Counter([obsv.count(True) for obsv in obsv_data if obsv.count(True)])
    pos_obsv_keys = list(pos_obsv_counter.keys())
    pos_obsv_keys.sort()
    pos_obsv = {i: pos_obsv_counter[i] for i in pos_obsv_keys}
    return pos_obsv

def get_people_gaze_info(track_data):
    observ_data = []
    positive_counts = []
    total_counts = []
    for rec_data in track_data:
        for person in rec_data:
            observ_data.append(person["observed"])
            positive_counts.append(person["observed"].count(True))
            total_counts.append(len(person["observed"]))
    positive_seconds = int(sum(positive_counts) / 30)
    total_seconds = int(sum(total_counts) / 30)
    print(f"# positive gaze matchings: {sum(positive_counts)} frames (~{positive_seconds} seconds).")
    print(f"# total gaze matchings: {sum(total_counts)} frames (~{total_seconds} seconds).")
    print(f"# positive different tracking times: {len(np.unique(positive_counts))}.")
    print(f"# total different tracking times: {len(np.unique(total_counts))}.")
    print(f"# different people detected: {len(observ_data)}.")
    return observ_data

def get_relevant_data(track_data, min_frames = 15):
    rel_people = []
    for rec_data in track_data:
        for person in rec_data:
            if person["observed"].count(True) >= min_frames:
                rel_people.append(person)
    return rel_people

if __name__ == "__main__":
    nx, ny = 700, 300
    #df_design = read_ds_design()
    #compute_gaze_areas_distribution(df_design)
    #gaze_list = compute_gaze_matrices(nx, ny)
    #plot_gaze_groups(gaze_list, df_design, nx, ny)
    track_data = read_tracking_data(rec_ids = range(1, 75))
    #get_people_gaze_info(track_data)
    rel_data = get_relevant_data(track_data, min_frames = 10)
    print(len(rel_data))
