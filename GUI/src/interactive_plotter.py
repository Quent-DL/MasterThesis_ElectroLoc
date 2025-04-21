"""This class defines an interactive PyVista viewer"""

from mediator_interface import MediatorInterface

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from pyvista.plotting.opts import ElementType

# Needed to plot an initially empty centroids mesh
pv.global_theme.allow_empty_mesh = True


class InteractivePlotter:
    def __init__(self):
        self._plotter = QtInteractor()

        # Adding callback for when a point is selected
        self._plotter.enable_point_picking(self.callback_pick, 
                                           show_message=False,
                                           picker='point', 
                                           use_picker=True,
                                           show_point=False)
        
        # Initializing the centroids (plotting empty points cloud)
        self._centroids_mesh = pv.PolyData(np.zeros((0,3), dtype=float))
        self._centroids_actor = self._plotter.add_mesh(self._centroids_mesh)
        
        # Preparing the selection sphere
        self._selection_mesh = pv.PolyData(np.zeros((1,3), dtype=float))
        self._selection_actor = self._plotter.add_mesh(
            self._selection_mesh, color=(255, 0, 0), point_size=12.5, 
            render_points_as_spheres=True, pickable=False)
        self._selection_actor.visibility = False

        # Adding some info text on how to move and pick, for the user
        self._plotter.add_text("Left click: orient cam\n"
                               "Right click: select point\n"
                               "Shift + Left click: move cam\n"
                               "Scroll: zoom",
                               font_size=8,
                               font='courier')

        # TODO store actors (centroids + CT)

        # TODO DEBUG REMOVE
        """ctds = np.indices((2,2,2), dtype=np.float32).reshape(3, -1).T
        self.replot_centroids(ctds)"""

    def get_widget(self) -> QtInteractor:
        return self._plotter

    def add_mediator(self, mediator: MediatorInterface) -> None:
        self._mediator = mediator

    def replot_centroids(self, centroids: np.ndarray) -> None:
        """Clears the previous centroids, and plot the given ones instead.
        
        ### Input:
        - centroids: the coordinates of the 3D centroids to display. 
        Shape (N, 3)."""

        # Storing the centroids for later use 
        # (e.g. updating or removing a centroid)
        self._centroids_mesh = pv.PolyData(centroids)

        # Replacing the actor, and removing the previous one
        old_actor = self._centroids_actor
        self._centroids_actor = self._plotter.add_mesh(
            self._centroids_mesh, color=(0, 0, 255), point_size=5.0, 
            render_points_as_spheres=True, pickable=True)
        self._plotter.remove_actor(old_actor)

        self._plotter.render()

    def add_centroid(self, coords: np.ndarray) -> None:
        """Adds a centroid at the specified coordinates (shape (3,))."""
        res = np.append(self._centroids_mesh.points, coords[np.newaxis], axis=0)
        self.replot_centroids(res)

    def remove_centroid(self, index: int) -> None:
        """Removes the centroid with the specified index"""
        res =  np.delete(self._centroids_mesh.points, index, axis=0)
        self.replot_centroids(res)
    
    def update_centroid(self, index: int, new_coords: np.ndarray) -> None:
        """Updates the position of one centroid.
        
        ### Inputs:
        - index: the id of the centroid
        - new_coords: its new coordinates. Shape (3,)."""
        self._centroids_mesh.points[index] = new_coords
        self._centroids_actor.mapper.Update()

    def select(self, index: int) -> None:
        self._selection_mesh.points[0] = self._centroids_mesh.points[index]
        self._selection_actor.visibility = True

    def unselect(self) -> None:
        self._selection_actor.visibility = False

    def callback_pick(self, actor, picker):
        centroid_id = picker.GetPointId()
        self._mediator.select_centroid(centroid_id)