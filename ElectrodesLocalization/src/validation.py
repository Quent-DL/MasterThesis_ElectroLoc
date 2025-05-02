"""This file's purpose is to provide metrics to validate the algorithm's 
results."""

from utils import distance_matrix

import numpy as np
import pandas as pd
from typing import Callable, Optional

def get_ground_truth(
        csv_path: str,
        transform_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """Returns the ground truth in the given CSV file. The file must contain
    columns 'ct_vox_x', 'ct_vox_y', 'ct_vox_z'.
    
    ### Inputs:
    - csv_path: the path to the CSV file. The file can contain comments
    that start with the character '#'.
    - transform_func: the transformation applied to each contact in the ground
    truth (e.g. a function that apply the affine transform from voxel space
    to physical space). Both input and outputs must be arrays of shape (N, 3).
    
    ### Output:
    - ground_truth: the coordinates of the contacts in the CSV.""" 
    # Extracting numpy array from CSV
    df_GT = pd.read_csv(csv_path, comment="#")
    ground_truth = df_GT[['ct_vox_x', 'ct_vox_y', 'ct_vox_z']].to_numpy(
        dtype=np.float32)
    if transform_func is not None:
        ground_truth = transform_func(ground_truth)
    return ground_truth


def validate_contacts_position(
        detected_contacts: np.ndarray, 
        ground_truth: np.ndarray,
        max_match_dist: float
) -> str:
    """TODO write documentation
    In: matrices of shape (N, 3) and (M, 3) in same coordinates space
    Out: None"""
    # Notation:
    # - dt: one single detected contact
    # - gt: one single ground truth contact
    # - DT: set of several (or all) dt's
    # - GT: set of several (or all) gt's

    # Shape (len(detected_contacts), len(ground_truth))
    distances = distance_matrix(detected_contacts, ground_truth)

    # Flag for each index to indicate if valid.
    # Initially, valid == dt and gt are close enough
    valid = distances < 0.5 * max_match_dist 

    # Indices of the matched dt's and gt's
    matched_dt_idx = []
    matched_gt_idx = []

    # Sorting indices by increasing distance, expressed in ravel format
    indices = distances.flatten().argsort()
    
    for i in indices:
        dt, gt = np.unravel_index(i, distances.shape)
        if valid[dt, gt]:
            # (dt, gt) is valid and of closest distance => match
            matched_dt_idx.append(dt)
            matched_gt_idx.append(gt)
            # Invalidating row dt and column gt, because each contact can
            # only match once
            valid[dt,:] = False
            valid[:,gt] = False
    
    # Finding missing contacts (= holes) (False Negative)
    # A missing contact happens when a gt is not matched to any dt
    holes_gt_idx = np.setdiff1d(np.arange(len(ground_truth)), matched_gt_idx)

    # Finding excess contacts (False Positive)
    # An excess contact happens when a dt is not matchd to any gt
    excess_dt_idx = np.setdiff1d(np.arange(len(detected_contacts)), matched_dt_idx)

    # Computing the distance between the matched (= relevant) contacts
    rlvt_distances = distances[np.array(matched_dt_idx), np.array(matched_gt_idx)]

    stat_prints = ("\n"
        "==================================================\n"
        "STATISTICAL RESULTS\n"
        "----------------[ CONTACTS ]----------------------\n"
        f"Contacts found: {len(detected_contacts)}"
        f"    ({len(matched_dt_idx)} matched, {len(excess_dt_idx)} in excess)\n"
        f"Missing       : {len(holes_gt_idx)}\n"
        "\n"
        "-------------[ DISTANCE ERROR ]-------------------\n"
        f"Mean: {rlvt_distances.mean():<9.3f}    "
            f"Std: {rlvt_distances.std():<9.3f}\n"
        f"Min:  {rlvt_distances.min():<9.3f}    "
            f"Max: {rlvt_distances.max():<9.3f}\n"
        f"25%:  {np.quantile(rlvt_distances, 0.25):<9.3f}    "
            f"95%: {np.quantile(rlvt_distances, 0.95):<9.3f}\n"
        "\n"
        "==================================================\n")
    
    return (matched_dt_idx, matched_gt_idx, excess_dt_idx, 
            holes_gt_idx, stat_prints)
