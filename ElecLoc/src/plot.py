"""
This file is responsible for plotting the data and models
"""

from electrode_models import (ElectrodeModel, 
                              LinearElectrodeModel, 
                              SegmentElectrodeModel,
                              ParabolicElectrodeModel)

import pyvista as pv
import random as random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Optional, Tuple
from copy import copy

# A constant to add to any point/point cloud so that it
# is rendered at the center of the voxels
VOX_CENTERING = np.array([0.5, 0.5, 0.5])

_COLOR_PALETTE = [
    
    (128, 128, 0),     # ???
    (0, 0, 255),       # blue
    (255, 150, 0),     # orange
    (0, 255, 255),     # cyan
    (255, 0, 0),       # red
    (255, 0, 255),     # purple
    (255, 255, 0),     # yellow
    (255, 230, 180),   # cream

    (120, 0, 0),       # dark pink
    (236, 78, 32),     # flame
    (28, 58, 19),      # pakistan green,

    (0, 114, 178),
    #(86, 180, 233),
    (240, 228, 66),
    (213, 94, 0),
    (204, 121, 167),
    (51, 117, 56),
    (221, 221, 221),
    (0, 156, 115),
]

def get_color(i: int) -> tuple:
    # Choosing color, or new random color if there are many electrodes
    color = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]     
    return color 


class ElectrodePlotter:
    """A wrapper class for plotting CT volumes and labelled points."""

    def __init__(self, func_world2vox: Callable[[np.ndarray], np.ndarray]):
        """Creates an instance of this class.
        
        ### Input:
        - func_world2vox: a function with a numpy array of shape (N, 3) for
        both the input and the output. Must also have an optional boolean 
        parameter 'apply_translation'."""
        self.func_world2vox = func_world2vox
        self.plotter = pv.Plotter()
        self.plotter.add_axes()

    def update_focal_point(self, focal_pt: np.ndarray,
                           convert_to_vox: bool=False) -> None:
        if convert_to_vox:
            focal_pt = self.func_world2vox(focal_pt)
        self.plotter.camera.focal_point = focal_pt


    def show(self) -> None:
        """Displays all the meshes, point clouds and volumes plotted."""
        self.plotter.show()
        
    def plot_contacts(
            self, 
            contacts: np.ndarray, 
            color: Optional[Tuple[int]] = None,
            size_multiplier: Optional[float] = 1,
            convert_to_vox: bool = False) -> None:
        """TODO write documentation"""
        if len(contacts) == 0:
            return
        if convert_to_vox:
            contacts = self.func_world2vox(contacts)
        point_cloud = pv.PolyData(contacts+VOX_CENTERING)
        self.plotter.add_points(
            point_cloud, 
            point_size=5.0*size_multiplier, 
            color=color,
            render_points_as_spheres=True)

    def plot_ct(self, ct: np.ndarray, alpha_factor: int = 1) -> None:
        """TODO write documentation"""
        # TODO Remove: shortcut for synthetic data
        if ct.max() <= 0:
            return
        vol = pv.ImageData()
        vol.dimensions = np.array(ct.shape) + 1
        vol.cell_data['values'] = ct.flatten(order='F')
        self.plotter.add_volume(vol, cmap="gray", opacity=[0,0.045*alpha_factor/5])

    def plot_ct_electrodes(self, ct_mask: np.ndarray, **kwargs) -> None:
        """TODO write documentation.
        
        Kwargs: cmap, opacity, and others from pyvista's Plotter.add_mesh"""
        # TODO Remove: synthetic data
        if ct_mask.max() <= 0:
            return
        
        kwargs['cmap'] = kwargs.get('cmap', 'Blues')
        kwargs['opacity'] = kwargs.get('opacity', 0.075)

        mesh_ct = pv.wrap(ct_mask)
        mesh_ct.cell_data['intensity'] = ct_mask[:-1, :-1, :-1].flatten(order='F')
        vol = mesh_ct.threshold(value=1, scalars='intensity')
        self.plotter.add_mesh(vol, scalars='intensity', **kwargs)
        self.plotter.remove_scalar_bar()

    def plot_colored_contacts(self, contacts: np.ndarray, 
                              labels: np.ndarray,
                              convert_to_vox:bool = False) -> None:
        """TODO write documentation"""
        if convert_to_vox:
            contacts = self.func_world2vox(contacts)
        # Iterate over each electrode and add its contacts to the plotter
        for k, e_id in enumerate(np.unique(labels)):
            color = get_color(k)
            point_cloud = pv.PolyData(contacts[labels == e_id])
            self.plotter.add_points(
                point_cloud+VOX_CENTERING, color=color, point_size=8, 
                render_points_as_spheres=True)

    def plot_differences(self, matched_DT, matched_GT,
                         convert_to_vox: bool=False) -> None:
        """TODO write documentation"""
        if convert_to_vox:
            matched_DT = self.func_world2vox(matched_DT)
            matched_GT = self.func_world2vox(matched_GT)
        for dt, gt in zip(matched_DT, matched_GT):
            line = pv.Line(
                dt + VOX_CENTERING, 
                gt + VOX_CENTERING)
            self.plotter.add_mesh(line, color=(0, 0, 0), line_width=1)

    def plot_electrodes_models(self, models: List[ElectrodeModel]) -> None:
        if isinstance(models[0], LinearElectrodeModel):
            self.plot_linear_electrodes(models)
        elif isinstance(models[0], ParabolicElectrodeModel):
            self.plot_parabolic_electrodes(models)
        elif isinstance(models[0], SegmentElectrodeModel):
            self.plot_segment_electrodes(models)

    def plot_linear_electrodes(
            self, models: List[LinearElectrodeModel],
            convert_to_vox: bool=True) -> None:
        for k, model in enumerate(models):
            color = get_color(k)
            if convert_to_vox:
                p = self.func_world2vox(model.point)
                v = self.func_world2vox(model.direction, apply_translation=False)
            else:
                p = model.point
                v = model.direction
            # TODO replace hard-coded 50 by meaningful values
            a, b = p - 50*v, p + 50*v
            line = pv.Line(a+VOX_CENTERING, b+VOX_CENTERING)
            self.plotter.add_mesh(line, color=color, line_width=3)

    def plot_parabolic_electrodes(
            self, models: List[ParabolicElectrodeModel],
            convert_to_vox: bool=True) -> None:
        for k, model in enumerate(models):
            # Creating a voxel-space copy of the model
            model_plot = copy(model)
            v, u, c = model.coefs.T
            if convert_to_vox:
                v = self.func_world2vox(v, apply_translation=False)
                u = self.func_world2vox(u, apply_translation=False)
                c = self.func_world2vox(c)
            model_plot.coefs = np.stack([v, u, c], axis=-1)

            # Plotting
            color = get_color(k)
            # TODO replace hard-coded 50 by meaningful value
            t = np.linspace(-50, 50, 100)
            x = model_plot.compute_position_at_t(t)
            spline = pv.Spline(x + VOX_CENTERING)
            self.plotter.add_mesh(spline, color=color, line_width=3)

    def plot_segment_electrodes(
            self, models: List[SegmentElectrodeModel],
            convert_to_vox: bool=True) -> None:
        for k, model in enumerate(models):
            color = get_color(k)
            if convert_to_vox:
                p = self.func_world2vox(model.point)
                v = self.func_world2vox(model.direction, apply_translation=False)
            else:
                p = model.point 
                v = model.direction
            a, b = p + model.t_a*v, p + model.t_b*v
            line = pv.Line(a+VOX_CENTERING, b+VOX_CENTERING)
            self.plotter.add_mesh(line, color=color, line_width=3)