from matplotlib import pyplot as plt
from drisafe.constants import DOWNTOWN_REC_IDS
from collections import Counter
from drisafe import stats

def plot_pos_gaze_track_counts(rec_ids1, rec_ids2, frame_min = 1, label1 = "",
                               label2 = "", show = False, save = False):
    fontsize = 12
    labelsize = 10
    xsize, ysize = 8, 6
    track_data1 = stats.read_tracking_data(rec_ids1)
    observ_data1 = stats.get_people_gaze_info(track_data1)
    pos_gaze_count1 = Counter([obsv.count(True) for obsv in observ_data1 if obsv.count(True) >= frame_min])
    sort_pos_keys1 = list(pos_gaze_count1.keys())
    sort_pos_keys1.sort()
    sort_pos_count1 = {i: pos_gaze_count1[i] for i in sort_pos_keys1}
    names1 = sort_pos_count1.keys()
    values1 = sort_pos_count1.values()
    track_data2 = stats.read_tracking_data(rec_ids2)
    observ_data2 = stats.get_people_gaze_info(track_data2)
    pos_gaze_count2 = Counter([obsv.count(True) for obsv in observ_data2 if obsv.count(True) >= frame_min])
    sort_pos_keys2 = list(pos_gaze_count2.keys())
    sort_pos_keys2.sort()
    sort_pos_count2 = {i: pos_gaze_count2[i] for i in sort_pos_keys2}
    names2 = sort_pos_count2.keys()
    values2 = sort_pos_count2.values()
    fig1, ax1 = plt.subplots(figsize=(xsize, ysize))
    ax1.bar([str(n) for n in names1], values1, log = True, color = "red", 
            edgecolor = "black", label = label1)
    ax1.bar([str(n) for n in names2], values2, log = True, color = "darkgreen", 
            edgecolor = "black", label = label2)
    ax1.grid(which = "both")
    ax1.set_xticklabels([f"{i / 30:.2f}" for i in names1], rotation = 90)
    ax1.set_xlabel("time [s]", fontsize = fontsize)
    ax2 = ax1.twiny()
    ax2.set_xticks(ax1.get_xticks())
    ax1.tick_params(axis = "both", labelsize = labelsize)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([f"{i}" for i in names1], rotation = 90)
    ax2.set_xlabel("frames [units]", fontsize = fontsize)
    ax1.set_ylabel("tracked people [units]", fontsize = fontsize)
    ax2.tick_params(axis = "both", labelsize = labelsize)
    fig1.tight_layout()
    ax1.legend()
    if save:
        fig1.savefig("results/plots/positive_gaze_track.png", dpi = 1000)
    if show:
        plt.show()

def plot_gaze_track_counts_cumul(rec_ids1, rec_ids2, frame_min = 1, label1 = "", 
                                 label2 = "", show = False, save = False):
    fontsize = 12
    labelsize = 10
    xsize, ysize = 8, 6
    track_data1 = stats.read_tracking_data(rec_ids1)
    observ_data1 = stats.get_people_gaze_info(track_data1)
    pos_gaze_count1 = Counter([obsv.count(True) for obsv in observ_data1 if obsv.count(True) >= frame_min])
    sort_pos_keys1 = list(pos_gaze_count1.keys())
    sort_pos_keys1.sort()
    sort_pos_count1 = {i: pos_gaze_count1[i] for i in sort_pos_keys1}
    track_data2 = stats.read_tracking_data(rec_ids2)
    observ_data2 = stats.get_people_gaze_info(track_data2)
    pos_gaze_count2 = Counter([obsv.count(True) for obsv in observ_data2 if obsv.count(True) >= frame_min])
    sort_pos_keys2 = list(pos_gaze_count2.keys())
    sort_pos_keys2.sort()
    sort_pos_count2 = {i: pos_gaze_count2[i] for i in sort_pos_keys2}
    great_gaze_counts1 = []
    great_gaze_counts2 = []
    for lim in sort_pos_keys1:
        val_obsv = [obsv for obsv in observ_data1 if obsv.count(True) >= lim]
        great_gaze_counts1.append(len(val_obsv))
    for lim in sort_pos_keys2:
        val_obsv = [obsv for obsv in observ_data2 if obsv.count(True) >= lim]
        great_gaze_counts2.append(len(val_obsv))
    fig1, ax1 = plt.subplots(figsize=(xsize, ysize))
    ax1.bar([str(n) for n in sort_pos_count1], great_gaze_counts1, log = False, 
            color = "red", edgecolor = "black", label = label1)
    ax1.bar([str(n) for n in sort_pos_count2], great_gaze_counts2, log = False, 
            color = "darkblue", edgecolor = "black", label = label2)
    ax1.grid(which = "both")
    ax1.set_xticklabels([f"{i / 30:.2f}" for i in sort_pos_count1], rotation = 90)
    ax1.set_xlabel("time [s]", fontsize = fontsize)
    ax2 = ax1.twiny()
    ax2.set_xticks(ax1.get_xticks())
    ax1.tick_params(axis = "both", labelsize = labelsize)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(sort_pos_count1, rotation = 90)
    ax2.set_xlabel("frames [units]", fontsize = fontsize)
    ax1.set_ylabel("tracked people [units]", fontsize = fontsize)
    ax2.tick_params(axis = "both", labelsize = labelsize)
    fig1.tight_layout()
    ax1.legend()
    if save:
        fig1.savefig("results/plots/cumulate_gaze_track.png", dpi = 1000)
    if show:
        plt.show()


if __name__ == "__main__":
    plot_pos_gaze_track_counts(rec_ids1 = range(1, 75), rec_ids2 = DOWNTOWN_REC_IDS, label1 = "All dataset", 
                               label2 = "Downtown", show = True, save = True)
    plot_gaze_track_counts_cumul(rec_ids1 = range(1, 75), rec_ids2 = DOWNTOWN_REC_IDS, label1 = "All dataset",
                                 label2 = "Downtown", show = True, save = True)