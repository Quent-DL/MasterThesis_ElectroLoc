import numpy as np

class DataCenter:
    """This class stores all the information that can be reused in the
    several panels of the application: centroids, CT, mask, electrodes, ..."""

    def __init__(self):
        self._centroids = np.empty((0,3))

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