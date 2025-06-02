"""This file's purpose is to provide metrics to validate the algorithm's 
results."""

import pipeline
import postprocessing
import linear_modeling
from misc.utils import match_and_swap_labels, distance_matrix, estimate_intercontact_distance
from misc.dataframe_contacts import DataFrameContacts
from misc.nib_wrapper import NibCTWrapper
from misc.electrode_information import ElectrodesInfo

import numpy as np
from typing import Tuple
import os


SUB_IDS = [
    "sub04",
    "sub11"
]


def get_data() -> list[tuple[NibCTWrapper, ElectrodesInfo, DataFrameContacts]]:
    data = []
    for subId in SUB_IDS:
        data_dir          = (f"{os.path.dirname(__file__)}/../data/{subId}/")
        elec_info_path    = os.path.join(data_dir, "in/electrodes_info.csv")
        ground_truth_path = (f"{os.path.dirname(__file__)}/../data_ground_truths/"
                             f"{subId}/ground_truth.csv")

        nib_wrapper = NibCTWrapper(
            os.path.join(data_dir, "in/CT.nii.gz"), 
            os.path.join(data_dir, "in/CTMask.nii.gz"))
        elec_info = ElectrodesInfo(elec_info_path)
        ground_truth = DataFrameContacts.from_csv(ground_truth_path)

        data.append((nib_wrapper, elec_info, ground_truth))

    return data


###
### Check position
###

def check_contacts_positions(
        detected_contacts: np.ndarray, 
        ground_truth: np.ndarray,
        max_match_dist: float
) -> Tuple[np.ndarray]:
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
    valid = distances < max_match_dist 

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
    
    # The arrays of pairs of indices of all the contacts matched to a ground truth
    if len(matched_dt_idx) != 0:
        matched_dt_idx = np.array(matched_dt_idx, dtype=int)
        matched_gt_idx = np.array(matched_gt_idx, dtype=int)
    else:
        matched_dt_idx = np.zeros((0,), dtype=int)
        matched_gt_idx = np.zeros((0,), dtype=int)

    # Finding missing contacts (= holes) (False Negative)
    # A missing contact happens when a gt is not matched to any dt
    holes_gt_idx = np.setdiff1d(np.arange(len(ground_truth)), matched_gt_idx)

    # Finding excess contacts (False Positive)
    # An excess contact happens when a dt is not matchd to any gt
    excess_dt_idx = np.setdiff1d(np.arange(len(detected_contacts)), matched_dt_idx)

    # Computing the distance between the matched (= relevant) contacts
    rlvt_distances = distances[matched_dt_idx, matched_gt_idx]

    return (matched_dt_idx, matched_gt_idx,
            excess_dt_idx, holes_gt_idx, 
            rlvt_distances)

def print_position_results(
        nb_detected: int,
        nb_expected: int,
        nb_matched: int,
        nb_excess: int,
        nb_missed: int,
        rlvt_distances: np.ndarray
) -> None:
    precision = 100*nb_matched/(nb_matched+nb_excess)
    recall = 100*nb_matched/(nb_matched+nb_missed)
    print("\n"
        "==================================================\n"
        "CONTACTS POSITIONS RESULTS\n"
        "----------------[ CONTACTS ]----------------------\n"
        f"False discovery rate:  {100-precision:.3f} % (-> Precision: {precision:.3f} %) \n"
        f"Miss rate:             {100-recall:.3f} % (-> Recall: {recall:.3f} %)\n"
        f"F1-Score:              {2*precision*recall / (precision + recall):.3f} %"
        "\n"
        f"Total contacts expected: {nb_expected}\n"
        f"Total contacts detected: {nb_detected} ({nb_matched} matched, {nb_excess} in excess)\n"
        f"Total contacts missed:   {nb_missed}\n"
        "-------------[ DISTANCE ERROR ]-------------------\n"
        f"Mean: {rlvt_distances.mean():<9.3f}    "
            f"Std: {rlvt_distances.std():<9.3f}\n"
        f"Min:  {rlvt_distances.min():<9.3f}    "
            f"Max: {rlvt_distances.max():<9.3f}\n"
        f"25%:  {np.quantile(rlvt_distances, 0.25):<9.3f}    "
            f"75%: {np.quantile(rlvt_distances, 0.75):<9.3f}\n"
        "\n"
        "(Only distances between matched pairs of detected and expected contacts used)\n"
        "==================================================\n\n")

def batch_validate_contacts_position():
    MATCH_DIST_TOLERANCE = 0.75

    data = get_data()

    nb_detected = 0
    nb_expected = 0
    nb_matched  = 0
    nb_excess   = 0
    nb_missed   = 0
    rlvt_distances = []

    for nib_wrapper, elec_info, ground_truth in data:
        output, _  = pipeline.pipeline(nib_wrapper, elec_info,
                                       print_logs=False)

        contacts_world = output.get_world_coordinates()
        
        ground_truth_world = nib_wrapper.convert_vox_to_world(
            ground_truth.get_vox_coordinates())

        intercontact_dist_world = estimate_intercontact_distance(
            ground_truth_world)

        results = check_contacts_positions(
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

    print_position_results(
        nb_detected, nb_expected, nb_matched, 
        nb_excess, nb_missed, rlvt_distances)

###
### Check classification
###

def batch_validate_classification():
    data = get_data()

    all_pred_labels   = []
    all_pred_labels_post = []
    all_expected_labels = []

    for nib_wrapper, elec_info, ground_truth in data:
        contacts_world = nib_wrapper.convert_vox_to_world(
            ground_truth.get_vox_coordinates())

        # Before preprocessing
        pred_labels, models = linear_modeling.classify_centroids(
            contacts_world,
            ground_truth[ground_truth._TAG_DCC_KEY].to_numpy(dtype=int),
            elec_info.nb_electrodes
        )

        # After preprocessing
        _, pred_labels_post, _, models_post = postprocessing.postprocess(
            contacts_world, 
            pred_labels, 
            models,
            elec_info,
            model_recomputation_class=None,
            # Parameters for validation: do not modify the contacts
            do_recompute_contacts=False,
            do_reorder_contacts=False,
            do_merge_models=True
        )

        # TODO DEBUG remove ###########
        from plot import ElectrodePlotter
        plotter = ElectrodePlotter(lambda x, apply_translation=False: x)
        plotter.plot_colored_contacts(contacts_world, pred_labels_post)
        plotter.plot_electrodes_models(models_post)
        plotter.show()

        ##############################

        # Retrieving the expected labels, and the matching predicted labels
        expected_labels = ground_truth.get_labels()
        pred_labels = match_and_swap_labels(
            expected_labels, pred_labels)
        pred_labels_post = match_and_swap_labels(
            expected_labels, pred_labels_post)

        all_pred_labels.append(pred_labels)
        all_pred_labels_post.append(pred_labels_post)
        all_expected_labels.append(expected_labels)
    
    all_pred_labels      = np.concatenate(all_pred_labels)
    all_pred_labels_post = np.concatenate(all_pred_labels_post)
    all_expected_labels  = np.concatenate(all_expected_labels)

    print_classification_results(all_pred_labels,
                                 all_pred_labels_post,
                                 all_expected_labels)

def print_classification_results(
        all_pred_labels: np.ndarray[int],
        all_pred_labels_post: np.ndarray[int],
        all_expected_labels: np.ndarray[int]
) -> None:
    accuracy = lambda all_pred: (np.sum(all_pred == all_expected_labels) 
                    / len(all_expected_labels) * 100)
    print("\n"
        "==================================================\n"
        "CENTROIDS CLASSIFICATION RESULTS\n"
        "----------------[ Metrics ]-----------------------\n"
        "Accuracy:\n"
        f" - Before postprocessing: {accuracy(all_pred_labels):.3f} % \n"
        f" - After postprocessing:  {accuracy(all_pred_labels_post):.3f} % \n"
        "==================================================\n\n")

###
### General
###

 
def main():
    batch_validate_classification()
    batch_validate_contacts_position()


if __name__ == '__main__':
    main()