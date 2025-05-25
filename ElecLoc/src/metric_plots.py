import pipeline
import postprocessing
import classification_cc
import validation
from utils import (distance_matrix, NibCTWrapper, ElectrodesInfo, 
                   estimate_intercontact_distance, PipelineOutput,
                   match_and_swap_labels)

import numpy as np
import pandas as pd
from typing import Callable, Optional, Tuple
import os
import matplotlib.pyplot as plt
import pickle


FORCE_RECOMPUTE = False
OUTPUT_DIR = "/etinfo/users/2023/qdelaet/MasterThesisSrc/z_metrics"

def exists(filename):
    return os.path.exists(os.path.join(OUTPUT_DIR), filename)

def dump(filename, data):
    with open(filename, "w") as f:
        pickle.dump(data, os.path.join(OUTPUT_DIR, filename))

def load(filename):
    with open(filename, "r") as f:
        pickle.load(os.path.join(OUTPUT_DIR, filename))




def metrics_distances():

    SAVE_FILE = "kde_distance_dt_gt"
    MATCH_DIST_TOLERANCE = 20000 #0.75

    data = validation.get_data()

    nb_detected = 0
    nb_expected = 0
    nb_matched  = 0
    nb_excess   = 0
    nb_missed   = 0
    rlvt_distances = []

    for nib_wrapper, elec_info, ground_truth in data:
        output, _  = pipeline.pipeline(nib_wrapper, elec_info,
                                       print_logs=False)

        if FORCE_RECOMPUTE or not exists(SAVE_FILE):
            contacts_world = output.get_world_coordinates()
            
            ground_truth_world = nib_wrapper.convert_vox_to_world(
                ground_truth.get_vox_coordinates())

            intercontact_dist_world = estimate_intercontact_distance(
                ground_truth_world)

            results = validation.check_contacts_positions(
                contacts_world, 
                ground_truth_world,
                MATCH_DIST_TOLERANCE*intercontact_dist_world)
            
            (matched_dt_idx, matched_gt_idx,
                excess_dt_idx, holes_gt_idx, 
                matched_dists) = results
            
            nb_detected += len(contacts_world)
            nb_expected += len(ground_truth)
            nb_matched += len(matched_dt_idx)
            nb_excess += len(excess_dt_idx)
            nb_missed += len(holes_gt_idx)
            rlvt_distances.append(matched_dists)
            rlvt_distances = np.concatenate(rlvt_distances)

            dump(SAVE_FILE, [nb_detected, nb_expected, nb_matched, nb_excess, nb_missed, rlvt_distances])

        else:
            nb_detected, nb_expected, nb_matched, nb_excess, nb_missed, rlvt_distances = load(SAVE_FILE)


    validation.print_position_results(
        nb_detected, nb_expected, nb_matched, 
        nb_excess, nb_missed, rlvt_distances)
    
    print_distances_kde(rlvt_distances)
    

def print_distances_kde(distances, sigma=0.05):
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(distances, 0.1)
    
    x = np.linspace(0, distances.max(), 2000)
    y = kde(x)
    cumulated = y.copy()
    tot = y.sum()
    for i, y_val in enumerate(y[1:]):
        cumulated[i] = cumulated[-1] + y_val / tot

    ax = plt.subplot()
    ax.set_xlabel("Distance [mm]")
    ax.set_ylabel("Density [mm^-1]")
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    metrics_distances()