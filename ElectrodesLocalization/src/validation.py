"""This file's purpose is to provide metrics to validate the algorithm's 
results."""

from utils import distance_matrix

import numpy as np
import pandas as pd
from typing import Callable


def __validate_contacts_position(
        detected_contacts: np.ndarray, 
        ground_truth: np.ndarray,
        transform_gt: Callable[[np.ndarray], np.ndarray]=None
) -> str:
    """TODO write documentation
    In: matrices of shape (N, 3) and (M, 3)
    transform_gt is a mapping function to re-express ground_truth coordinates
    into detected_contacts coordinates space
    Out: None"""
    if transform_gt:
        # Expressing ground truth in same coordinates space as detected contacts
        ground_truth = transform_gt(ground_truth)

    distances_global = distance_matrix(detected_contacts, ground_truth)

    # indices of the matched and single contacts (both DT or GT)
    matched_DT = []
    matched_GT = []
    single_DT  = np.arange(len(detected_contacts))
    single_GT  = np.arange(len(ground_truth))

    iter_nb = 0
    keep_looping = True

    # Keep looping as long as matches are found
    while keep_looping:
        # Compute mean and std of current matches
        if iter_nb != 0:
            matched_distances = distances_global[np.array(matched_DT), np.array(matched_GT)]
            mtch_mean = matched_distances.mean()
            mtch_std = matched_distances.std()

        # Compute distance matrix between non-matched (= single) contacts
        distances_singles = distances_global[np.ix_(
            np.array(single_DT), np.array(single_GT))]
        closest_GT_to_DT = distances_singles.argmin(axis=1)
        closest_DT_to_GT = distances_singles.argmin(axis=0)

        # -- First round of matches --
        # Finding mutual matches between the still-single DT and GT
        # A match happens when two single dt and gt 
        # are mutually closest to each other AND within reasonable distance
        keep_looping = False
        for single_idx, global_idx in enumerate(single_GT):
            if closest_GT_to_DT[closest_DT_to_GT[single_idx]] == single_idx:
                # dt and gt mutually closest among single contacts
                if iter_nb == 0 or distances_singles[closest_DT_to_GT[single_idx], single_idx] < mtch_mean + 5*mtch_std:
                    # dt and gt are within reasonable distance
                    matched_GT.append(global_idx)
                    matched_DT.append(single_DT[closest_DT_to_GT[single_idx]])
                    keep_looping = True

        # Updating list of contacts (gt and dt) that are still single
        single_GT = set(range(len(ground_truth))).difference(matched_GT)
        single_GT = np.array(list(single_GT))

        single_DT = set(range(len(detected_contacts))).difference(matched_DT)
        single_DT = np.array(list(single_GT))

        iter_nb += 1
    
    # Finding missing contacts (False Negative)
    # A missing contact happens when a gt is not matched to any dt
    missing = set(range(len(ground_truth))).difference(matched_GT)

    # Finding excess contacts (False Positive)
    # An excess contact happens when, for a dt, 
    # its closest gt is already matched
    excess = set(range(len(detected_contacts))).difference(matched_DT)

    # Computing the distance between the actually matched contacts (= relevant)
    rlvt_distances = distances_global[np.array(matched_DT), np.array(matched_GT)]

    stat_prints = ("\n"
        "==================================================\n"
        "STATISTICAL RESULTS\n"
        "----------------[ CONTACTS ]----------------------\n"
        f"Contacts found: {len(detected_contacts)}"
        f"    ({len(matched_DT)} matched, {len(excess)} excess)\n"
        f"Missing       : {len(missing)}\n"
        "\n"
        "-------------[ DISTANCE ERROR ]-------------------\n"
        f"Mean: {rlvt_distances.mean():<9.3f}    "
            f"Std: {rlvt_distances.std():<9.3f}\n"
        f"Min:  {rlvt_distances.min():<9.3f}    "
            f"Max: {rlvt_distances.max():<9.3f}\n"
        f"25%:  {np.quantile(rlvt_distances, 0.25):<9.3f}    "
            f"75%: {np.quantile(rlvt_distances, 0.75):<9.3f}\n"
        "\n"
        "==================================================\n")
    
    return matched_DT, matched_GT, excess, missing, stat_prints


def validate_contacts_position_from_file(
        detected_contacts: np.ndarray, 
        ground_truth_filepath: str,
        transform_gt: Callable[[np.ndarray], np.ndarray]=None
) -> str:
    
    """TODO write documentation
    In: matrices of shape (N, 3), a path
    transform_gt is a mapping function to re-express ground_truth coordinates
    into detected_contacts coordinates space
    Out: None"""
    # Extracting numpy array from CSV
    df_GT = pd.read_csv(ground_truth_filepath, comment="#")
    ground_truth = df_GT[['ct_vox_x', 'ct_vox_y', 'ct_vox_z']].to_numpy(
        dtype=np.float32)
    
    return __validate_contacts_position(
        detected_contacts, 
        ground_truth,
        transform_gt)

