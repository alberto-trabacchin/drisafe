from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
from drisafe import stats

def plot_people_gaze_counts():
    fontsize = 10
    labelsize = 8
    xsize, ysize = 11, 6
    track_data = stats.read_tracking_data()
    observ_data = stats.get_people_gaze_info(track_data)
    pos_gaze_count = Counter([obsv.count(True) for obsv in observ_data])
    sort_pos_keys = list(pos_gaze_count.keys())
    sort_pos_keys.sort()
    sort_pos_count = {i: pos_gaze_count[i] for i in sort_pos_keys[1::]}
    names = sort_pos_count.keys()
    values = sort_pos_count.values()
    fig1, ax1 = plt.subplots(figsize=(xsize, ysize))
    ax1.bar([str(n) for n in names], values, log = True, color = "darkgreen", 
            edgecolor = "black")
    ax1.grid(axis = "x", which = "both")
    ax1.set_xticklabels([f"{i / 30:.2f}" for i in names], rotation = 90)
    ax1.set_xlabel("time [s]", fontsize = fontsize)
    ax2 = ax1.twiny()
    ax2.set_xticks(ax1.get_xticks())
    ax1.tick_params(axis = "both", labelsize = labelsize)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([f"{i}" for i in names], rotation = 90)
    ax2.set_xlabel("frames [units]", fontsize = fontsize)
    ax1.set_ylabel("tracked people [units]", fontsize = fontsize)
    ax2.tick_params(axis = "both", labelsize = labelsize)
    fig1.tight_layout()
    fig1.savefig("results/plots/positive_people_track.png", dpi = 1000)

    tot_gaze_count = Counter([len(obsv) for obsv in observ_data if obsv.count(True) >= 1])
    sort_tot_keys = list(tot_gaze_count.keys())
    sort_tot_keys.sort()
    sort_tot_count = {i: tot_gaze_count[i] for i in sort_tot_keys[1::]}
    names = sort_tot_count.keys()
    values = sort_tot_count.values()
    fig2, ax3 = plt.subplots(figsize=(xsize, ysize))
    ax3.bar([str(n) for n in names], values, log = True, color = "darkblue", 
            edgecolor = "black")
    ax3.set_xticklabels([f"{i / 30:.2f}" for i in names], rotation = 90)
    ax3.set_xlabel("time [s]", fontsize = fontsize)
    ax4 = ax3.twiny()
    ax4.set_xticks(ax3.get_xticks())
    ax3.tick_params(axis = "both", labelsize = labelsize)
    ax4.set_xbound(ax3.get_xbound())
    ax4.set_xticklabels(names, rotation = 90)
    ax4.set_xlabel("frames [units]", fontsize = fontsize)
    ax3.set_ylabel("tracked people [units]", fontsize = fontsize)
    ax4.tick_params(axis = "both", labelsize = labelsize)
    fig2.tight_layout()
    fig2.savefig("results/plots/total_people_track.png", dpi = 1000)

    great_gaze_counts = []
    for lim in sort_pos_keys[1::]:
        val_obsv = [obsv for obsv in observ_data if obsv.count(True) >= lim]
        great_gaze_counts.append(len(val_obsv))
    fig3, ax5 = plt.subplots(figsize=(xsize, ysize))
    ax5.bar([str(n) for n in sort_pos_count], great_gaze_counts, log = False, 
            color = "darkblue", edgecolor = "black")
    ax5.grid(axis = "x", which = "both")
    ax5.set_xticklabels([f"{i / 30:.2f}" for i in sort_pos_count], rotation = 90)
    ax5.set_xlabel("time [s]", fontsize = fontsize)
    ax6 = ax5.twiny()
    ax6.set_xticks(ax5.get_xticks())
    ax3.tick_params(axis = "both", labelsize = labelsize)
    ax6.set_xbound(ax5.get_xbound())
    ax6.set_xticklabels(sort_pos_count, rotation = 90)
    ax6.set_xlabel("frames [units]", fontsize = fontsize)
    ax5.set_ylabel("tracked people [units]", fontsize = fontsize)
    ax6.tick_params(axis = "both", labelsize = labelsize)
    fig3.tight_layout()
    fig3.savefig("results/plots/total_people_track_integral.png", dpi = 1000)
    plt.show()




if __name__ == "__main__":
    plot_people_gaze_counts()