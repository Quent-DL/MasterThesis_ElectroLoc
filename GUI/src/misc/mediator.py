"""The implementation of the interface described in mediator.py"""

from misc.mediator_interface import MediatorInterface
from extraction_panel import ElecLocExtractionPanel
from interactive_plotter import InteractivePlotter
from misc.data_center import DataCenter

import numpy as np

class Mediator(MediatorInterface):
    def __init__(
            self, 
            extraction_panel: ElecLocExtractionPanel,
            interactive_plotter: InteractivePlotter,
            data_center: DataCenter
    ):
        self._extraction_panel = extraction_panel
        self._interactive_plotter = interactive_plotter
        self._dc = data_center

        self._index = -1

    def _hasSelected(self) -> bool:
        return (self._index != -1 
                and self._index < self._dc.nb_centroids())
    
    def _getSelected(self) -> np.ndarray:
        if not self._hasSelected():
            raise RuntimeError("No centroid currently selected")
        return self._dc.get_centroids()[self._index]

    def set_centroids(self, centroids):
        self._dc.set_centroids(centroids)
        self._interactive_plotter.replot_centroids(centroids)

        if len(centroids) > 0:
            self.select_centroid(0)
    
    def add_centroid(self):
        if self._hasSelected():
            coords = self._getSelected() + np.array([0,0,1])
        else:
            coords = self._dc.get_centroids().mean(axis=0)      # Shape (3,).

        # Adding new centroid
        self._dc.add_centroid(coords)
        self._interactive_plotter.add_centroid(coords)

        # Selecting this new centroid
        self.select_centroid(self._dc.nb_centroids()-1)

    def delete_selected_centroid(self):
        # Cannot remove a nonexistant centroid
        if not self._hasSelected():
            return
        
        # Deleting centroid from local list
        self._dc.delete_centroid(self._index)
        self._interactive_plotter.delete_centroid(self._index)
        
        self.unselect_centroid()

    def update_selected_centroid(self, new_coords):
        self._dc.update_centroid(self._index, new_coords)
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
            self._getSelected())
    
    def unselect_centroid(self):
        self._index = -1
        self._interactive_plotter.unselect()
        self._extraction_panel.unselect()

    
    def replot_ct(self, ct, opacity):
        self._interactive_plotter.plot_input_ct(ct, opacity)