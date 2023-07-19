from matplotlib import pyplot as plt
from drisafe.constants import DOWNTOWN_REC_IDS
from collections import Counter
from drisafe import stats
 
def get_tracking_counts(all_rec_ids, down_rec_ids):
    all_track = stats.read_tracking_data(all_rec_ids)
    down_track = stats.read_tracking_data(down_rec_ids)
    all_obsv = stats.get_obsv_data(all_track)
    down_obsv = stats.get_obsv_data(down_track)
    all_pos_obsv = stats.count_obsv_data(all_obsv)
    down_pos_obsv = stats.count_obsv_data(down_obsv)
    all_ped_counts = list(all_pos_obsv.values())
    down_ped_counts = list(down_pos_obsv.values())
    all_time_counts = list(all_pos_obsv.keys())
    down_time_counts = list(down_pos_obsv.keys())
    for i in range(len(all_time_counts)):
        if all_time_counts[i] != down_time_counts[i]:
            down_time_counts.insert(i, all_time_counts[i])
            down_ped_counts.insert(i, 0)
    all_obsv = dict(zip(all_time_counts, all_ped_counts))
    down_obsv = dict(zip(down_time_counts, down_ped_counts))
    return all_obsv, down_obsv

def plot_tracking_counts(all_rec_ids, down_rec_ids, show = True, save = False):
    fontsize = 12
    labelsize = 10
    xsize, ysize = 8, 6
    all_obsv, down_obsv = get_tracking_counts(all_rec_ids, down_rec_ids)
    fig, ax1 = plt.subplots(figsize=(xsize, ysize))
    y1 = list(all_obsv.values())
    x = list(map(str, all_obsv.keys()))
    y2 = list(down_obsv.values())
    ax1.bar(x, y1, log = True, color = "red", edgecolor = "black", label = "All dataset")
    ax1.bar(x, y2, log = True, color = "darkgreen", edgecolor = "black", label = "Downtown")
    ax1.grid(which = "both")
    ax1.set_xticklabels([f"{int(i) / 30:.2f}" for i in x], rotation = 90)
    ax1.set_xlabel("time [s]", fontsize = fontsize)
    ax2 = ax1.twiny()
    ax2.set_xticks(ax1.get_xticks())
    ax1.tick_params(axis = "both", labelsize = labelsize)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([f"{int(i)}" for i in x], rotation = 90)
    ax2.set_xlabel("frames [units]", fontsize = fontsize)
    ax1.set_ylabel("tracked people [units]", fontsize = fontsize)
    ax2.tick_params(axis = "both", labelsize = labelsize)
    fig.tight_layout()
    ax1.legend()
    if save:
        fig.savefig("results/plots/positive_gaze_track.png", dpi = 1000)
    if show:
        plt.show()    

def plot_tracking_cumul(all_rec_ids, down_rec_ids, show = True, save = False):
    fontsize = 12
    labelsize = 10
    xsize, ysize = 8, 6
    all_obsv, down_obsv = get_tracking_counts(all_rec_ids, down_rec_ids)
    all_ped_counts = list(all_obsv.values())
    x = list(map(str, all_obsv.keys()))
    down_ped_counts = list(down_obsv.values())
    y1_cumul = []
    y2_cumul = []
    for i in range(len(all_ped_counts)):
        y1_cumul.append(sum(all_ped_counts[i::]))
        y2_cumul.append(sum(down_ped_counts[i::]))
    fig, ax1 = plt.subplots(figsize=(xsize, ysize))
    ax1.bar(x, y1_cumul, log = False, color = "red", edgecolor = "black", label = "All dataset")
    ax1.bar(x, y2_cumul, log = False, color = "darkblue", edgecolor = "black", label = "Downtown")
    ax1.grid(which = "both")
    ax1.set_xticklabels([f"{int(i) / 30:.2f}" for i in x], rotation = 90)
    ax1.set_xlabel("time [s]", fontsize = fontsize)
    ax2 = ax1.twiny()
    ax2.set_xticks(ax1.get_xticks())
    ax1.tick_params(axis = "both", labelsize = labelsize)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([f"{int(i)}" for i in x], rotation = 90)
    ax2.set_xlabel("frames [units]", fontsize = fontsize)
    ax1.set_ylabel("tracked people [units]", fontsize = fontsize)
    ax2.tick_params(axis = "both", labelsize = labelsize)
    fig.tight_layout()
    ax1.legend()
    if save:
        fig.savefig("results/plots/cumulate_gaze_track.png", dpi = 1000)
    if show:
        plt.show()


if __name__ == "__main__":
    plot_tracking_counts(all_rec_ids = range(1, 75), down_rec_ids = DOWNTOWN_REC_IDS,
                         save = True, show = False)
    plot_tracking_cumul(all_rec_ids = range(1, 75), down_rec_ids = DOWNTOWN_REC_IDS,
                        save = True, show = False)