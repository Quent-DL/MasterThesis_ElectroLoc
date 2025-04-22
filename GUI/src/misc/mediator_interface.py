"""This class represents an interface for a mediator whose goal
is to maintain communication between the Extraction Panel
and the Interactive Plotter about the centroids, electrodes, ...
so that manually changing info in the former updates the latter,
and selecting in the latter displays information in the former."""

from abc import ABC, abstractmethod
import numpy as np

class MediatorInterface(ABC):
    @abstractmethod
    def set_centroids(self, centroids: np.ndarray) -> None:
        pass

    @abstractmethod
    def add_centroid(self) -> None:
        pass

    @abstractmethod
    def delete_selected_centroid(self) -> None:
        pass

    @abstractmethod
    def update_selected_centroid(self, new_coords: np.ndarray) -> None:
        pass

    @abstractmethod
    def select_centroid(self, index: int) -> None:
        pass

    @abstractmethod
    def unselect_centroid(self) -> None:
        pass

    @abstractmethod
    def load_plot_ct_volumes(self, ct_path: str, mask_path: str) -> bool:
        """Tries to load the given Nifti files. Returns True if the
        operation is successful, and False otherwise."""
        pass

    @abstractmethod
    def update_ct_display(self, visibility: bool, opacity: float) -> None:
        pass

    
    @abstractmethod
    def plot_thresholded_volume(
            self, threshMin: float, threshMax: float,
            visibility: bool, opacity: float) -> None:
        """Plots a new volume with the given threshold"""
        pass

    @abstractmethod
    def update_thresholded_display(
            self, visibility: bool, opacity: float) -> None:
        """Difference with plot_thresholded_volume: only affects
        visibility and opacity, not thresholds"""
        pass
