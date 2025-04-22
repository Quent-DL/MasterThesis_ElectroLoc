"""This class defines an interactive PyVista viewer"""

from misc.mediator_interface import MediatorInterface

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from pyvista.plotting.opts import ElementType
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction

# Needed to plot an initially empty centroids mesh
pv.global_theme.allow_empty_mesh = True


class InteractivePlotter:
    def __init__(self):
        self._plotter = QtInteractor()
        self._plotter.add_axes()

        # Adding callback for when a point is picked
        # -> delegating to mediator
        self._plotter.enable_point_picking(
            lambda _, pckr: self._mediator.select_centroid(pckr.GetPointId()),
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
            self._selection_mesh, color=(255, 0, 0), point_size=8.5, 
            render_points_as_spheres=True, pickable=False)
        self._selection_actor.visibility = False

        # Adding some info text on how to move and pick, for the user
        self._plotter.add_text("Left click: orient cam\n"
                               "Right click: select point\n"
                               "Shift + Left click: move cam\n"
                               "Scroll: zoom",
                               font_size=8,
                               font='courier')
        
        # Initializing the grayscale input (CT) (plotting empty volume)
        # TODO keep or remove
        """self._ct_actor = self._plotter.add_volume(
            np.array([[[0, 1]]], dtype=float))
        self._ct_actor.visibility = False"""
        self._ct_actor = pv.Actor()
        self._thresholded_actor = pv.Actor()

        # TODO store actors (centroids + CT)

    def get_widget(self) -> QtInteractor:
        return self._plotter

    def add_mediator(self, mediator: MediatorInterface) -> None:
        self._mediator = mediator

    #
    # Centroids
    #

    def replot_centroids(self, centroids: np.ndarray) -> None:
        """Clears the previous centroids, and plot the given ones instead.
        
        ### Input:
        - centroids: the coordinates of the 3D centroids to display. 
        Shape (N, 3)."""

        # Storing the centroids for later use 
        # (e.g. updating or removing a centroid)
        
        # increment by 0.5 to align points and volumes
        centroids = centroids + 0.5

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

    def delete_centroid(self, index: int) -> None:
        """Removes the centroid with the specified index"""
        res =  np.delete(self._centroids_mesh.points, index, axis=0)
        self._centroids_mesh.points = res
        self._centroids_actor.mapper.Update()

        #self.replot_centroids(res)
    
    def update_centroid(self, index: int, new_coords: np.ndarray) -> None:
        """Updates the position of one centroid.
        
        ### Inputs:
        - index: the id of the centroid
        - new_coords: its new coordinates. Shape (3,)."""
        new_coords = new_coords + 0.5    # to align points and volumes
        self._centroids_mesh.points[index] = new_coords
        self._centroids_actor.mapper.Update()
        self._selection_mesh.points[0] = new_coords
        self._selection_actor.mapper.Update()

    def select(self, index: int) -> None:
        self._selection_mesh.points[0] = self._centroids_mesh.points[index]
        self._selection_actor.visibility = True
        self._selection_actor.mapper.Update()

    def unselect(self) -> None:
        self._selection_actor.visibility = False
        self._selection_actor.mapper.Update()

    #
    # CT volumes
    #

    def plot_input_ct(self, masked_ct: np.ndarray) -> None:
        """Plots the given CT volume and removes the previous one 
        (if it exists). Input shape (L, W, H). """

        """ TODO keep this version or version below
        vol = pv.ImageData()
        vol.dimensions = np.array(ct.shape) + 1
        vol.cell_data['values'] = ct.flatten(order='F')
        self.plotter.add_volume(vol, cmap="gray", opacity=[0,0.045/5])"""

        old_actor = self._ct_actor
        self._ct_actor = self._plotter.add_volume(
            masked_ct, cmap='gray', 
            # opacity=[0.0, opacity],     TODO keep or remove
            show_scalar_bar=False, pickable=False)
        self._plotter.remove_actor(old_actor)

        self.ct_min = masked_ct.min()
        self.ct_max = masked_ct.max()

        self._plotter.render()   # TODO see if useful

    def update_ct_display(self, visibility: bool, opacity: float) -> None:
        # TODO visibility doesn't work
        self._ct_actor.visibility = visibility
        #self._ct_actor.prop.SetOpacity(opacity)    TODO remove
        self._change_volume_opacity(
            self._ct_actor, self.ct_min, self.ct_max, opacity)
        self._ct_actor.mapper.Update()

    def plot_thresholded(self, thresholded_mask: np.ndarray) -> None:
        """Plots the given binary mask volume and removes the previous one 
        (if it exists). Input shape (L, W, H). """

        mesh = pv.wrap(thresholded_mask)
        mesh.cell_data['intensity'] = thresholded_mask[:-1, :-1, :-1].flatten(order='F')
        vol = mesh.threshold(value=1, scalars='intensity')

        old_actor = self._thresholded_actor
        # TODO remove scalar bar
        self._thresholded_actor = self._plotter.add_mesh(
            vol, cmap='Blues', scalars='intensity', 
            # opacity=0.075     TODO keep or remove
            )
        self._plotter.remove_actor(old_actor)

        self._plotter.render()    # TODO see if useful

    def update_thresholded_display(self, visibility: bool, opacity: float) -> None:
        self._thresholded_actor.visibility = visibility
        self._thresholded_actor.GetProperty().opacity = opacity
        #self._thresholded_actor.mapper.Update()

    # TODO remove useless function (only used once)
    def _change_volume_opacity(self, actor: pv.Actor, 
                               min_vol: float, max_vol: float, opacity: float):
        #opacity_function = vtkPiecewiseFunction()
        #opacity_function.AddPoint(min_vol, 0.0)
        #opacity_function.AddPoint(max_vol, opacity)

        # TODO debug remove
        actor.GetProperty().opacity_unit_distance = opacity