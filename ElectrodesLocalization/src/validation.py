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

    distances = distance_matrix(detected_contacts, ground_truth)
    closest_GT_to_DT = distances.argmin(axis=1)
    closest_DT_to_GT = distances.argmin(axis=0)

    # Finding matches between DT and GT
    # A match happens when a dt and a gt are mutually the closest to each other
    matched_DT = []
    matched_GT = []
    for i, _ in enumerate(ground_truth):
        if closest_GT_to_DT[closest_DT_to_GT[i]] == i:
            matched_GT.append(i)
            matched_DT.append(closest_DT_to_GT[i])
    
    # Finding missing contacts (False Negative)
    # A missing contact happens when a gt is not matched to any dt
    missing = set(range(len(ground_truth))).difference(matched_GT)

    # Finding excess contacts (False Positive)
    # An excess contact happens when, for a dt, 
    # its closest gt is already matched
    excess = set(range(len(detected_contacts))).difference(matched_DT)

    # Computing the distance between the actually matched contacts (= relevant)
    rlvt_distances = distances[np.array(matched_DT), np.array(matched_GT)]

    stat_prints = ("\n"
        "==================================================\n"
        "STATISTICAL RESULTS\n"
        "----------------[ CONTACTS ]----------------------\n"
        f"Contacts found: {len(detected_contacts)}"
        f"    ({len(matched_DT)} matched, {len(excess)} excess)\n"
        f"Missing       : {len(missing)}\n"
        "\n"
        "--------[ DISTANCE OF MATCHED PAIRS ]-------------\n"
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

