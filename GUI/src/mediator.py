"""The implementation of the interface described in mediator.py"""

from mediator_interface import MediatorInterface
from extraction_panel import ElecLocExtractionPanel
from interactive_plotter import InteractivePlotter

import numpy as np

class Mediator(MediatorInterface):
    def __init__(
            self, 
            extraction_panel: ElecLocExtractionPanel,
            interactive_plotter: InteractivePlotter):
        self._extraction_panel = extraction_panel
        self._interactive_plotter = interactive_plotter

        self._centroids = np.empty((0,3), dtype=np.float32)
        self._index = -1

        # TODO debug remove
        """ctds = np.indices((2,2,2), dtype=np.float32).reshape(3, -1).T
        self.set_centroids(ctds)"""

    def _hasSelected(self) -> bool:
        return self._index != -1 and self._index < len(self._centroids)
    
    def _getSelected(self) -> np.ndarray:
        if not self._hasSelected():
            raise RuntimeError("No centroid currently selected")
        return self._centroids[self._index]

    def set_centroids(self, centroids):
        self._centroids = centroids
        self._interactive_plotter.replot_centroids(centroids)

        if len(centroids) > 0:
            self.select_centroid(0)

    def get_centroids(self):
        return self._centroids.copy()
    
    def add_centroid(self):
        if self._hasSelected():
            coords = self._getSelected() + np.array([0,0,1])
        else:
            coords = self._centroids.mean(axis=0)         # Shape (3,).

        self._centroids = np.concatenate([self._centroids, 
                                          coords[np.newaxis]])
        self._interactive_plotter.add_centroid(coords)
        self.select_centroid(len(self._centroids)-1)

    def delete_selected_centroid(self):
        # Cannot remove a nonexistant centroid
        if not self._hasSelected():
            return
        
        # Deleting centroid from local list
        self._centroids = np.delete(self._centroids, self._index, axis=0)

        # Removing centroid from plotter
        self._interactive_plotter.remove_centroid(self._index)

        """ TODO keep or delete
        # Selecting "next" centroid (which is now at current index)
        # Accounts for the case where the list is now empty (will select index -1)
        self.select_centroid(min(self._index, len(self._centroids)-1))"""
        self.unselect_centroid()

    def update_selected_centroid(self, new_coords):
        self._centroids[self._index] = new_coords
        self._interactive_plotter.update_centroid(self._index, new_coords)

    def select_centroid(self, index):
        # Ignore duplicate selection
        if self._index == index:
            return
        
        self._index = index

        # Special case: selecting -1 means unselecting
        if index == -1:
            self.unselect_centroid()
            return

        # Highlights the centroid in the viewer
        self._interactive_plotter.select(index)
        # Displays the centroid's coordinates in the panel
        self._extraction_panel.display_selected_centroid(
            self._centroids[index])
    
    def unselect_centroid(self):
        self._index = -1
        self._interactive_plotter.unselect()
        self._extraction_panel.unselect()

    
    def replot_ct(self, ct, opacity):
        # TODO ESSENTIAL
        raise NotImplementedError()

