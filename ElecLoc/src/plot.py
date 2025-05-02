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


__COLOR_PALETTE = [
    
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
    if i < len(__COLOR_PALETTE):
        color = __COLOR_PALETTE[i]
    else:
        # random electrode color
        color = [random.randint(0,255) for _ in range(3)]       
    return color 


class ElectrodePlotter:
    """A wrapper class for plotting CT volumes and labelled points."""

    def __init__(self, func_world2vox: Callable[[np.ndarray], np.ndarray]):
        """Creates an instance of this class.
        
        ### Input:
        - func_world2vox: a function with a numpy array of shape (N, 3) for
        both the input and the output. Must also have an optional boolean 
        parameter 'apply_translation'."""
        self.func_world2vox=  func_world2vox
        self.plotter = pv.Plotter()
        self.plotter.add_axes()

    def update_focal_point(self, focal_pt_world: np.ndarray) -> None:
        self.plotter.camera.focal_point = self.func_world2vox(focal_pt_world)

    def show(self) -> None:
        """Displays all the meshes, point clouds and volumes plotted."""
        self.plotter.show()
        
    def plot_contacts(
            self, 
            contacts: np.ndarray, 
            color: Optional[Tuple[int]] = None,
            size_multiplier: Optional[float] = 1) -> None:
        """TODO write documentation"""
        if len(contacts) == 0:
            return
        point_cloud = pv.PolyData(self.func_world2vox(contacts))
        self.plotter.add_points(
            point_cloud, 
            point_size=5.0*size_multiplier, 
            color=color,
            render_points_as_spheres=True)

    def plot_ct(self, ct: np.ndarray) -> None:
        """TODO write documentation"""
        # TODO Remove: shortcut for synthetic data
        if ct.max() <= 0:
            return
        vol = pv.ImageData()
        vol.dimensions = np.array(ct.shape) + 1
        vol.cell_data['values'] = ct.flatten(order='F')
        self.plotter.add_volume(vol, cmap="gray", opacity=[0,0.045/5])

    def plot_ct_electrodes(self, ct_mask: np.ndarray) -> None:
        """TODO write documentation"""
        # TODO Remove: synthetic data
        if ct_mask.max() <= 0:
            return
        mesh_ct = pv.wrap(ct_mask)
        mesh_ct.cell_data['intensity'] = ct_mask[:-1, :-1, :-1].flatten(order='F')
        vol = mesh_ct.threshold(value=1, scalars='intensity')
        self.plotter.add_mesh(vol, cmap='Blues', scalars='intensity', 
                              opacity=0.075)

    def plot_colored_contacts(self, contacts: np.ndarray, 
                              labels: np.ndarray) -> None:
        """TODO write documentation"""
        contacts = self.func_world2vox(contacts)
        # Iterate over each electrode and add its contacts to the plotter
        for k, e_id in enumerate(np.unique(labels)):
            color = get_color(k)
            point_cloud = pv.PolyData(contacts[labels == e_id])
            self.plotter.add_points(
                point_cloud, color=color, point_size=8, 
                render_points_as_spheres=True)

    def plot_differences(self, matched_DT, matched_GT) -> None:
        """TODO write documentation"""
        for dt, gt in zip(matched_DT, matched_GT):
            line = pv.Line(self.func_world2vox(dt), self.func_world2vox(gt))
            self.plotter.add_mesh(line, color=(0, 0, 0), line_width=1)

    def plot_electrodes_models(self, models: List[ElectrodeModel]) -> None:
        if isinstance(models[0], LinearElectrodeModel):
            self.plot_linear_electrodes(models)
        elif isinstance(models[0], ParabolicElectrodeModel):
            self.plot_parabolic_electrodes(models)
        elif isinstance(models[0], SegmentElectrodeModel):
            self.plot_segment_electrodes(models)

    def plot_linear_electrodes(
            self, models: List[LinearElectrodeModel]) -> None:
        for k, model in enumerate(models):
            color = get_color(k)
            p = self.func_world2vox(model.point)
            v = self.func_world2vox(model.direction, apply_translation=False)
            # TODO replace 50 by maningful values
            a, b = p - 50*v, p + 50*v
            line = pv.Line(a, b)
            self.plotter.add_mesh(line, color=color, line_width=3)

    def plot_parabolic_electrodes(
            self, models: List[ParabolicElectrodeModel]) -> None:
        for k, model in enumerate(models):
            # Creating a voxel-space copy of the model
            model_plot = copy(model)
            v, u, c = model.coefs.T
            v = self.func_world2vox(v, apply_translation=False)
            u = self.func_world2vox(u, apply_translation=False)
            c = self.func_world2vox(c)
            model_plot.coefs = np.stack([v, u, c], axis=-1)

            # Plotting
            color = get_color(k)
            # TODO replace hard-coded 70 by meaningful value
            t = np.linspace(-50, 50, 100)
            x = model_plot.compute_position_at_t(t)
            spline = pv.Spline(x)
            self.plotter.add_mesh(spline, color=color, line_width=3)

    def plot_segment_electrodes(
            self, models: List[SegmentElectrodeModel]) -> None:
        for k, model in enumerate(models):
            color = get_color(k)
            p = self.func_world2vox(model.point)
            v = self.func_world2vox(model.direction, apply_translation=False)
            a, b = p + model.t_a*v, p + model.t_b*v
            line = pv.Line(a, b)
            self.plotter.add_mesh(line, color=color, line_width=3)