import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA

class Electrode:
    """TODO documentation"""
    
    def __init__(self, contacts: np.ndarray, ct_shape):
        """TODO write documentation"""
        ct_center = np.array(ct_shape, dtype=np.float32) / 2

        # Extracting the contacts that belong to electrode e and sorting them
        # by their value along the axis
        pca = PCA(n_components=1)
        scores = pca.fit_transform(contacts).ravel()   # ravel to convert (N,1) to (N,)
        sorted_contacts = contacts[np.argsort(scores)]

        # If necessary, reversing the order of the contact the electrode so 
        # that the first contact is the deepest
        # (i.e. the closest to the center of the ct)
        if norm(sorted_contacts[0]-ct_center) > norm(sorted_contacts[-1]-ct_center):
            # First contact is deeper than last => reverse the electrode
            sorted_contacts = np.flip(sorted_contacts, axis=0)

        self.contacts = sorted_contacts