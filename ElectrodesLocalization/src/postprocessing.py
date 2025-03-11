import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA


def __sort_indices_by_contact_depth(
        indices: np.ndarray, 
        contacts: np.ndarray, 
        center: np.ndarray
) -> np.ndarray:
    """Sorts the contacts indices of a single electrode by depth. 
    
    ### Inputs:
    - indices: an integer array of shape (K,) that contains the indices of
    the coordinates in 'contacts' to consider and sort. K must belong to 
    {0, ..., N-1} (N is defined in the description of 'contacts').
    - contacts: an array of shape (N, 3) that contains the 3D coordinates of
    all the contacts detected in the CT.
    - center: an array of shape (3,) that contains the coordinates of the 
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
    if norm(first_contact-center) > norm(last_contact-center):
        # First contact is deeper than last => reverse the electrode
        sorted_indices = np.flip(sorted_indices)

    return sorted_indices


def get_electrodes_contacts_ids(
        contacts: np.ndarray, 
        labels: np.ndarray,
        center: np.ndarray
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
    - center: an array of shape (3,) that contains the coordinates of the 
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
            elec_indices, contacts, center)
        nb_contacts = np.shape(elec_indices)[0]
        contacts_ids[sorted_indices] = np.arange(nb_contacts)

    # TODO debug remove
    assert not np.any(contacts_ids == -1) 

    return contacts_ids