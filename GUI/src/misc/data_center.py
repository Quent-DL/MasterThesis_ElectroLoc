from elecloc.utils import NibCTWrapper

import numpy as np
import os

class DataCenter:
    """This class stores all the information that can be reused in the
    several panels of the application: centroids, CT, mask, electrodes, ..."""

    def __init__(self):
        self._centroids = np.empty((0,3))
        self._ct_object = None

    #
    # Centroids
    #

    def get_centroids(self) -> np.ndarray:
        return self._centroids
    
    def nb_centroids(self) -> int:
        return len(self._centroids)

    def set_centroids(self, centroids: np.ndarray) -> None:
        """Sets the centroids. Shape (N, 3)."""
        self._centroids = centroids

    def add_centroid(self, coords: np.ndarray) -> None:
        """Adds one centroid to the stored ones. Shape (3,)"""
        self._centroids = np.append(
            self._centroids, coords[np.newaxis], axis=0)
        
    def delete_centroid(self, index: int) -> None:
        self._centroids = np.delete(self._centroids, index, axis=0)

    def update_centroid(self, index: int, new_coords: np.ndarray) -> None:
        self._centroids[index] = new_coords

    #
    # CT volumes
    #

    def try_load_ct_object(self, ct_path: str, mask_path: str) -> bool:
        """Returns whether the file(s) was/were succesfully loaded.
        Only changes the internal attribute in case of success."""
        ct_path = ct_path.strip()
        mask_path = mask_path.strip()

        # Mandatory: ct_path must be valid
        try:
            NibCTWrapper.test_path(ct_path)
        except (FileNotFoundError, ValueError):
            # If load failed, keep previous ct object
            return False
        
        # Optional: mask_path
        try:
            NibCTWrapper.test_path(mask_path, allow_empty=True)
        except (FileNotFoundError, ValueError):
            mask_path = None

        # Loading files
        self._ct_object = NibCTWrapper(ct_path, mask_path)
        return True

    def get_ct_object(self):
        return self._ct_object
        