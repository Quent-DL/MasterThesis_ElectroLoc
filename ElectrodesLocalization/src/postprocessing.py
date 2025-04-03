import utils
from utils import ElectrodesInfo
from electrode_models import ElectrodeModel, LinearElectrodeModel

import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from typing import Tuple, List, Type
from scipy.optimize import minimize


def __sort_indices_by_contact_depth(
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


def __get_electrodes_contacts_ids(
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
        sorted_indices = __sort_indices_by_contact_depth(
            elec_indices, contacts, ct_center)
        nb_contacts = np.shape(elec_indices)[0]
        contacts_ids[sorted_indices] = np.arange(nb_contacts)

    # TODO debug remove
    assert not np.any(contacts_ids == -1) 

    return contacts_ids


def  __reassign_labels_closest(
        contacts: np.ndarray, 
        labels: np.ndarray
) -> np.ndarray:
    """Matches the contacts to the closest electrode.
    TODO write documentation
    
    ### Inputs:
    - contacts: an array of shape (N, 3) that contains the 3D coordinates of
    all N contacts detected in the CT.
    - labels: an integer array of shape (N,) that contains, for each contact,
    the id of the electrode it has been classified into.
    
    ### Output:
    - result: an array of shape (N,) whose content is similar to 'labels', but
    has been re-adjusted to assign each contact to the closest electrode.
    """

    # TODO replace hyperparam
    HYPER_PARAM_DIST = 3

    distances = []

    for e_id in np.unique(labels):
        if e_id == -1:
            # Not re-assigning "unlabelled" state to contacts
            continue

        # Computing the linear regression of that electrode
        p_e, v_e = utils.get_regression_line_parameters(
            points=contacts[labels==e_id])
        # Computing distance between each contact and this regression
        # Src: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        # Shape (N,)
        distances_e = norm(np.cross(v_e, (p_e-contacts)), axis=1) / norm(v_e)
        distances.append(distances_e)
    
    # distances[i, j] = distance between contact i and regression of electrode j.
    # Shape (N_electrodes, N_contacts)
    distances = np.stack(distances)

    # TODO check if it works
    dists = distances.argmin(axis=0)
    to_update = (labels == -1) & (dists < HYPER_PARAM_DIST)
    print(f"{to_update.nonzero()[0].shape[0]} contacts reassigned out of {(labels==-1).nonzero()[0].shape[0]} ")
    labels[to_update] = dists[to_update]
    return labels
    

# TODO remove deprecated
def __merge_similar_electrodes(
        contacts: np.ndarray, 
        labels: np.ndarray, 
        n_electrodes: int
) -> np.ndarray:
    """TODO write documentation"""
    # TODO debug so that labels end up in range [0, n_electrodes)
    uniques = np.unique(labels)
    while len(uniques) > n_electrodes:
        inters, dirs = [], []
        for l in uniques:
            # Regress electrode-wise
            data = contacts[labels == l]    # shape (K, 3)  
            # Shapes (3,) and (3,)
            inter, dir = utils.get_regression_line_parameters(data)
            inters.append(inter[1:])
            dirs.append(dir[1:])
        inters = np.stack(inters)    # Shape (K, 2)
        dirs   = np.stack (dirs)

        # Project onto both planes (x-max and x-min)
        proj_min = inters + contacts[1,:].min() * dirs
        proj_max = inters + contacts[1,:].max() * dirs
        projs = np.concatenate([proj_min, proj_max], axis=1)    # Shape (K, 4)

        # Distance map
        distance_map = utils.distance_matrix(projs)
        n = len(uniques)
        distance_map[range(n), range(n)] = distance_map.max()

        # Selects two closest electrodes (= projections)
        i, j = np.unravel_index(distance_map.argmin(), distance_map.shape)

        # Merge similar electrodes
        li = uniques[i]
        lj = uniques[j]
        labels[labels == lj] = li
        uniques = np.unique(labels)


def __estimate_intercontact_distance(
        contacts: np.ndarray
) -> Tuple[float, float]:
    """Returns an estimate of the inter-contact distance based on an histogram.
    This function computes the distance matrix of the contacts, then computes
    an estimate of the average smallest distance between two contacts 
    (ignoring outliers with unnaturally small distances).
    
    ### Input:
    - contacts: an array of shape (N, 3) that contains the 3D coordinates of
    all the contacts detected in the CT.
    
    ### Outputs:
    - dist: an estimate of the inter-contact distance.
    - dist_std: the standard deviation of the distance between each contact
    and its closest neighbor."""
    # Distance matrix of the contacts. Shape (N, N)
    distance_map = utils.distance_matrix(contacts)
    # Ensuring that the closest detected neighbor of a contact isn't itself.
    distance_map[distance_map==0] = distance_map.max()

    # For each contact, the distance to its closest neighbor. Shape (N,)
    distances_neigh = distance_map.min(axis=1)

    # Identifying the mode of the histogram
    step = 0.2
    bins = np.arange(0, distances_neigh.max()+step, step)
    hist, _ = np.histogram(distances_neigh, bins)
    mode = hist.argmax()    # index of the modal bin

    # Applying mean to modal bin and neighboring bins (IF not on border)
    bin_min = max(0, mode-1)
    bin_max = min(len(bins)-1, mode+2)
    dist = distances_neigh[
        (bins[bin_min] < distances_neigh) 
        & (distances_neigh < bins[bin_max])].mean()

    return dist #, distances_neigh.std() TODO remove if useless

def __match_labels(
        entry_points: np.ndarray,
        old_models: List[ElectrodeModel],
        old_labels: np.ndarray
) -> np.ndarray:
    """TODO write documentation
    - len(models) == entry_points.shape[0] == len(old_labels)"""
    # Distances between each model and each entry point
    distances = []
    for k, model in enumerate(old_models):
        distances.append(model.compute_distance(entry_points))

    # Shape (n_models, n_entry_points)   (should normally be squared)
    distances = np.stack(distances)

    # The new labels such that, if model[i] matches entry_poins[j],
    # then i = convert_to_old[j]
    # TODO use other system to ensure double bijection between old and new labels
    convert_to_old = distances.argmin(axis=0)

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
    assert not -1 in new_labels, "Not all old labels have been updated"

    return new_models, new_labels


def __model_fit_loss(
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


def __model_fit_apply(
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
        loss = lambda t0: __model_fit_loss(
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
        new_labels.append(k * np.ones((nb_contacts[k],)))
        new_positions_ids.append(np.arange(nb_contacts[k]))
    
    new_contacts      = np.concatenate(new_contacts)
    new_labels        = np.concatenate(new_labels)
    new_positions_ids = np.concatenate(new_positions_ids)
    return new_contacts, new_labels, new_positions_ids


def postprocess(
        contacts: np.ndarray, 
        labels: np.ndarray,
        ct_center: np.ndarray,
        models: List[ElectrodeModel],
        elec_info: ElectrodesInfo,
        intercontact_distance: float=None,
        model_cls: Type[ElectrodeModel] = LinearElectrodeModel
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ElectrodeModel]:
    """TODO write documentation"""

    # TODO write rest of implementation of the function

    if intercontact_distance is None:
        intercontact_distance = __estimate_intercontact_distance(contacts)


    # Re-classify contacts using linear/quadratic regression
    # and assigning closest regression to each contact
    # TODO uncomment
    #labels = __reassign_labels_closest(contacts, labels)

    # ...

    # Re-fitting models with new class, based on support of previous models
    for k, model in enumerate(models):
        models[k] = model_cls(contacts[labels==k])

    # Computing the electrode-wise positional id of each contact
    positions_ids = __get_electrodes_contacts_ids(contacts, labels, ct_center)

    # Sorting contacts by electrode id, then positional id
    order = np.lexsort(keys=(positions_ids, labels))
    contacts      = contacts[order]
    labels        = labels[order]
    positions_ids = positions_ids[order]

    # Swapping labels ids to match nb of electrodes
    models, labels = __match_labels(elec_info.entry_points, models, labels)

    # TODO uncomment: Mapping contacts to points along the model
    contacts, labels, positions_ids = __model_fit_apply(
        models, contacts, labels, elec_info.nb_contacts, intercontact_distance)

    return contacts, labels, positions_ids, models