import utils
from utils import ElectrodesInfo
from electrode_models import (ElectrodeModel, SegmentElectrodeModel, 
                              LinearElectrodeModel, compute_sRsquared)

import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from typing import Tuple, List, Type
from scipy.optimize import minimize


def _merge_two_most_similar_models(
        models: list[SegmentElectrodeModel], 
        contacts: np.ndarray[float],
        labels: np.ndarray[int]
) -> tuple[list[SegmentElectrodeModel], np.ndarray[int]]:
    """Merges two models such that the s-R-squared score of group of models
    is the least impacted.

    ### Outputs:
    - new_models: the updated list with merged models, such that 
    len(new_models) = len(models) - 1.
    - new_labels: the new labels that match 'new_models'."""
    
    def _get_merged(idx_merged_a, idx_merged_b):
        """Returns a copy of the models and labels, but after merging
        two models."""
        new_models = []
        new_labels = np.empty_like(labels)
        # Adding non-merged models
        for i, model in enumerate(models):
            if i == idx_merged_a or i == idx_merged_b:
                continue
            new_labels[labels == i] = len(new_models)
            new_models.append(model)
        # Adding new merged model
        inliers_a_or_b = (labels == idx_merged_a) | (labels == idx_merged_b)
        new_labels[inliers_a_or_b] = len(new_models)
        new_models.append(SegmentElectrodeModel(contacts[inliers_a_or_b]))

        return new_models, new_labels
    
    # Computing the score of all possible ways of merging 2 models
    scores = []
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            new_models, new_labels = _get_merged(i, j)
            score = compute_sRsquared(new_models, contacts, new_labels)
            scores.append((new_models, new_labels, score))

    # Selecting the merged pair with the best score
    scores.sort(key=lambda item: item[2], reverse=True)
    best_models, best_labels, _ = scores[0]
    return best_models, best_labels
    

def _sort_indices_by_contact_depth(
        indices: np.ndarray, 
        contacts: np.ndarray, 
        ct_center: np.ndarray
) -> np.ndarray:
    """Sorts the contacts indices of a single electrode by depth. 
    
    ### Inputs:
    - indices: an integer array of shape (K,) that contains the indices of
    the coordinates in 'contacts' to consider and sort. K must belong to 
    {0, ..., N-1} (N is defined in the description of 'contacts').
    - contacts: an array of shape (N, 3) that contains the 3D coordinates of
    all the contacts detected in the CT.
    - ct_center: an array of shape (3,) that contains the coordinates of the 
    center of the CT, in the same coordinate system as that of 'contacts'.
    Is is used to determine which end of each electrode is the deepest.
    
    ### Returns:
    - sorted_indices: an array of shape (K,) that contains the same values as
    in indices, but sorted such that sorted_indices[0] refers to the deepest
    contact of the electrode, and sorted_indices[-1] refers to the contact that
    is closest to the electrode's entry point."""

    # Corner case: don't and can't compute PCA if only one contact in electrode
    # -> it's already sorted
    if indices.shape[0] <= 1:
        return indices

    # Sorting the relevant contacts by their value 
    # on the main axis of the electrode
    pca = PCA(n_components=1)
    # ravel below to convert (N,1) to (N,)
    scores = pca.fit_transform(contacts[indices]).ravel()   
    sorted_indices = indices[np.argsort(scores)]

    # If necessary, reversing the order of the contacts of the electrode so 
    # that the first contact of the array is the deeper of the two
    # (i.e. the closer to the center of the ct)
    first_contact = contacts[sorted_indices[0]]
    last_contact = contacts[sorted_indices[-1]]
    if norm(first_contact-ct_center) > norm(last_contact-ct_center):
        # First contact is deeper than last => reverse the electrode
        sorted_indices = np.flip(sorted_indices)

    return sorted_indices


def _get_electrodes_contacts_ids(
        contacts: np.ndarray, 
        labels: np.ndarray,
        ct_center: np.ndarray
) -> np.ndarray:
    """Returns the id of each contact along its associated electrode. 
    
    For example, suppose the electrode #2 contains 4 contacts located at 
    indices 19, 21, 56, 75 in the matrix 'contacts'. Also suppose that the 
    order of these contacts along the electrode, from deepest to closest to 
    entry point, is 56, 75, 19, 21 (contact 56 is the deepest of electrode #2, 
    and contact 21 is the entry point of electrode #2). Then, the output 
    'contacts_ids' will be such that contacts_ids[56] = 0, 
    contacts_ids[75] = 1, contacts_ids[19] = 2, and contacts_ids[21] = 3.
    This process is done for each electrode present in 'labels'.
    
    ### Inputs:
    - contacts: an array of shape (N, 3) that contains the 3D coordinates of
    all the contacts detected in the CT.
    - labels: an integer array of shape (N,) that contains, for each contact,
    the id of the electrode it has been classified into.
    - ct_center: an array of shape (3,) that contains the coordinates of the 
    center of the CT, in the same coordinate system as that of 'contacts'.
    Is is used to determine which end of each electrode is the deepest.
    
    ### Output:
    - contacts_ids: an array of shape (N,) that contains, for each contact,
    the position it occupies on its electrode. The position is encoded as an
    integer in range [0, nb_contacts-1]."""
    contacts_ids = - np.ones_like(labels)    # default id = -1
    for e_id in np.unique(labels):
        # index [0] below because nonzero returns a tuple of length 1
        elec_indices = np.nonzero(labels == e_id)[0]
        sorted_indices = _sort_indices_by_contact_depth(
            elec_indices, contacts, ct_center)
        nb_contacts = np.shape(elec_indices)[0]
        contacts_ids[sorted_indices] = np.arange(nb_contacts)

    # TODO debug remove
    assert not np.any(contacts_ids == -1) 

    return contacts_ids


def _match_labels_to_entry_points(
        entry_points: np.ndarray,
        old_models: List[ElectrodeModel],
        old_labels: np.ndarray
) -> np.ndarray:
    """TODO write documentation
    - len(models) == entry_points.shape[0] == len(old_labels)"""
    assert len(old_models) == len(entry_points), ("There should be an "
        "identical number of models and entry points.\n"
        f"Expected {len(entry_points)}. Got {len(old_models)}.\n"
        f"Please check your inputs.")
    # Distances between each model and each entry point
    distances = []
    for k, model in enumerate(old_models):
        distances.append(model.compute_distance(entry_points))
    # Shape (n_models, n_entry_points)   -> square matrix
    distances = np.stack(distances)

    # The new labels such that, if model[i] matches entry_points[j],
    # then i = convert_to_old[j]
    # Using stable marriage problem to find an optimal bijection between
    # old and new labels.
    convert_to_old = np.empty((len(old_models),), dtype=int)
    invalid = distances.max() + 1    # Unpickable value in min or argmin
    for _ in range(len(old_models)):
        # picking pair with smallest distance
        (i, j) = np.unravel_index(distances.argmin(), distances.shape)
        # making each member of the pair unavailable for the following picks
        distances[i,:] = invalid
        distances[:,j] = invalid
        convert_to_old[j] = i

    # Updating list of models
    # Source: https://stackoverflow.com/questions/6618515/sorting-list-according-to-corresponding-values-from-a-parallel-list

    # Updating labels
    new_labels = -1 * np.ones_like(old_labels)    # -1 is a placeholder
    new_models = []
    for j, i in enumerate(convert_to_old):
        # i is an old label, which must be replaced by j
        new_models.append(old_models[i])
        new_labels[old_labels == i] = j
    
    # TODO Debug, replace by handling the case and forcing the mapping to be bijective
    assert not -1 in new_labels, "Algo bug: Not all old labels have been updated"

    return new_models, new_labels


def _model_fit_loss(
        t0: float, 
        model: ElectrodeModel, 
        contacts: np.ndarray, 
        nb_points: int,
        intercontact_dist: float,
        gamma: int
) -> float:
    """TODO write documentation"""
    # Shape (N, 3)
    targets = model.get_sequence(nb_points, t0, intercontact_dist, gamma)
    # Distance between each point of the sequence, and its closest neighbor
    # among 'contacts'. Shape (nb_points, len(contacts)).
    distances = utils.distance_matrix(targets, contacts).min(axis=1)
    return np.sum(distances**2)


# TODO remove dependence on 'nb_contacts' -> make it optional
def _model_fit_apply(
        models: List[ElectrodeModel], 
        contacts: np.ndarray, 
        labels: np.ndarray,
        nb_contacts: List[int],
        intercontact_dist: float,
) -> np.ndarray:
    """TODO write documentation
    - Assumes electrode-wise contacts sorted by decreasing depth
    - Assumes values in 'labels' match with indices in 'nb_contacts'"""
    new_contacts      = []
    new_labels        = []
    new_positions_ids = []

    for k, model in enumerate(models):
        model_contacts = contacts[labels == k]
        gamma = model.get_gamma(model_contacts[0], model_contacts[-1])

        # Defining loss function
        loss = lambda t0: _model_fit_loss(
            t0[0], model, model_contacts, nb_contacts[k], 
            intercontact_dist, gamma)    
        
        # Defining t0 from which to start the search
        init_t0 = model.get_projection_t(model_contacts[0])    # float
        init_t0 = np.array([init_t0])    # Shape (1,) for scipy
        
        # Computing optimal t0* that locally minimizes loss
        # TODO: try replacing by minimize_scalar (even if no init_t0 can be given)
        optimize_res = minimize(loss, init_t0)
        if not optimize_res.success:
            print("Warning: Could not optimize model k."
                  "Using non-projected contacts instead.\n"
                  f"Reason: {optimize_res.message}\n")
            # Using old, unprojected contacts instead of a projected and
            # equidistant version
            new_contacts.append(model_contacts)
            new_labels.append(k * np.ones((len(model_contacts),)))
            new_positions_ids.append(np.arange(len(model_contacts)))
            continue
        
        t0: float = optimize_res.x[0]    

        # Getting sequence of points that starts at t0*
        model_sequence = model.get_sequence(
            nb_contacts[k], t0, intercontact_dist, gamma)
        new_contacts.append(model_sequence)
        new_labels.append(k * np.ones((nb_contacts[k],), dtype=int))
        new_positions_ids.append(np.arange(nb_contacts[k]))
    
    new_contacts      = np.concatenate(new_contacts)
    new_labels        = np.concatenate(new_labels)
    new_positions_ids = np.concatenate(new_positions_ids)
    return new_contacts, new_labels, new_positions_ids


def postprocess(
        contacts: np.ndarray, 
        labels: np.ndarray,
        ct_center: np.ndarray,
        models: List[SegmentElectrodeModel],
        elec_info: ElectrodesInfo,
        intercontact_distance: float=None,
        model_cls: Type[ElectrodeModel] = LinearElectrodeModel
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ElectrodeModel]:
    """TODO write documentation"""

    # TODO Handle
    # raise RuntimeError("TODO: assert that len(models) matches with elec_info, if given.")

    if intercontact_distance is None:
        intercontact_distance = utils.estimate_intercontact_distance(contacts)

    # ...

    # Checking nnumber of models, and adapting them if necessary
    print(f"Before: {len(models)}")    # TODO deubug remove
    if len(models) < elec_info.nb_electrodes:
        raise RuntimeError("Too few models received in postprocessing.\n"
                           f"Expected {len(elec_info.nb_contacts)}. Got {len(models)}.\n"
                           "Increase the classification score threshold then try again.")
    while len(models) > elec_info.nb_electrodes:
        models, labels = _merge_two_most_similar_models(
            models, contacts, labels)
    print(f"Before: {len(models)}")    # TODO debug remove

    # Re-fitting models with new class, based on support of previous models.
    # Computations are weighted to avoid overfitting.
    for k, model in enumerate(models):
        inliers_k = contacts[labels==k]
        dist = model.compute_distance(inliers_k)
        weights = np.exp(- ( dist / (intercontact_distance/2))**2)
        models[k] = model_cls(inliers_k, weights)

    # Computing the electrode-wise positional id of each contact
    positions_ids = _get_electrodes_contacts_ids(contacts, labels, ct_center)

    # Sorting contacts by electrode id, then positional id
    order = np.lexsort(keys=(positions_ids, labels))
    contacts      = contacts[order]
    labels        = labels[order]
    positions_ids = positions_ids[order]

    # Swapping labels ids to match nb of electrodes
    models, labels = _match_labels_to_entry_points(elec_info.entry_points, models, labels)

    # Mapping contacts to points along the model
    contacts, labels, positions_ids = _model_fit_apply(
        models, contacts, labels, elec_info.nb_contacts, intercontact_distance)

    return contacts, labels, positions_ids, models