from sklearn.cluster import KMeans
import numpy as np
from utils import Electrode
from typing import List


def __extract_features(contacts: np.ndarray) -> np.ndarray:
    """From the coordinates of the contacts, extracts and returns the features 
    used in the clustering technique.

    Input:
    - contacts: an array of shape (N, 3) that contains the coordinates of the N
    contacts to cluster.

    Output:
    - features: an array of shape (N, M) that contains the M features for each of
    the N contacts.
    
    The features for each contact c consist in:
    - the coordinates of c.
    - the vector from c to its closest contact, potentially multiplied by -1
    as to have a positive x component in the vector. This is done such that
    e.g. vectors (-1, 2, 3) and (1, -2, -3) are considered as the same."""

    # Computing the distance map of the contacts
    diff = contacts[:, np.newaxis, :] - contacts[np.newaxis, :, :]
    distance_map = np.sqrt(np.sum(diff**2, axis=-1))

    # Sabotaging the diagonal so that a contact cannot be closest to itself
    n = contacts.shape[0]
    distance_map[range(n), range(n)] = distance_map.max()

    # Computing the closest contact to each
    # closest[i] is the coordinates of the contact that is closest to contact i
    closest_indices = np.argmin(distance_map, axis=1)   # Shape (N,)
    closest = contacts[closest_indices]    # Shape (N, 3)

    # Computing the vector between each contact and its closest contact
    # and multiplying where needed to have a positive x component
    vector = contacts - closest
    vector_posx = np.where(vector[:,:1] >= 0, vector, -vector)

    # Concatenating the coordinates and vectors into the features
    features = np.concatenate([contacts, vector_posx], axis=1)

    return features


def segment_electrodes(
        contacts: np.ndarray,
        n_electrodes: int, 
        ct_shape: tuple
) -> List[Electrode]:
    """Groups contacts into electrodes.
    
    Inputs:
    - contacts: an array of shape (N, 3) that contains the coordinates of the
    N contacts to group into electrodes.
    - n_electrodes: the number of electrodes to group the contacts.
    - ct_shape: a tuple of length 3 that contains the dimensions. It is used to
    determine the direction of the electrode (which end is the deepest, and
    which end is at the entry of the skull).
    
    Outputs:
    electrodes: a list of electrodes computed from the given contacts. The list
    is of length n_electrodes."""
    # Feature extraction
    features = __extract_features(contacts)

    # Applying KMeans to retrieve 'labels', an array of shape (N,) that
    # contains the label of each contact (label is in range [0, n_electrodes))
    # The label is the id of the electrode
    kmeans = KMeans(n_clusters=n_electrodes)
    labels = kmeans.fit_predict(features)

    # Constructing the electrodes one by one
    electrodes = []
    for e in range(n_electrodes):
        relevant_contacts = contacts[labels == e]
        electrodes.append(Electrode(relevant_contacts, ct_shape))

    return electrodes