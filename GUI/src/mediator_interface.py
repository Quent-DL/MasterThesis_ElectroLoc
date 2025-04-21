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
    def get_centroids(self) -> np.ndarray:
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
    def replot_ct(self, ct: np.ndarray[float], opacity: float) -> None:
        pass