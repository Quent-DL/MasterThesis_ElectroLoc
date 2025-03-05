"""
This file is responsible for plotting the data
"""

from utils import Electrode
import pyvista as pv
import random as random
import numpy as np
from typing import List

__COLOR_PALETTE = [
    (0, 114, 178),
    #(86, 180, 233),
    (240, 228, 66),
    (213, 94, 0),
    (204, 121, 167),
    (0, 0, 0),
    (51, 117, 56),
    (221, 221, 221),
    (0, 156, 115),
]


def plot_colored_electrodes(
    electrodes: List[Electrode], 
    plotter: pv.Plotter = None
) -> pv.Plotter:
    """TODO write documentation"""

    if plotter is None:
        plotter = pv.Plotter()

    all_contacts = []

    # Iterate over each electrode and add its contacts to the plotter
    for i, e in enumerate(electrodes):
        all_contacts.append(e.contacts)
        
        # Choosing color, or new random color if there are many electrodes
        if i < len(__COLOR_PALETTE):
            color = __COLOR_PALETTE[i]
        else:
            color = [random.random() for _ in range(3)]        # random electrode color

        point_cloud = pv.PolyData(e.contacts)
        plotter.add_points(point_cloud, color=color, point_size=15.0, 
                        render_points_as_spheres=True)
    
    # Centers the camera around the center of electrodes
    mean = np.concatenate(all_contacts, axis=0).mean(axis=0)
    plotter.camera.focal_point = mean

    return plotter


def plot_binary_electrodes(
        ct_mask: np.ndarray,
        plotter: pv.Plotter = None
) -> pv.Plotter:
    """TODO write documentation"""
    if plotter is None:
        plotter = pv.Plotter()

    mesh_ct = pv.wrap(ct_mask)
    mesh_ct.cell_data['intensity'] = ct_mask[:-1, :-1, :-1].flatten(order='F')
    vol = mesh_ct.threshold(value=1, scalars='intensity')
    plotter.add_mesh(vol, cmap='Blues', scalars='intensity', opacity=0.1)
    return plotter

def plot_ct(
        ct: np.ndarray,
        plotter: pv.Plotter = None
) -> pv.Plotter:
    """TODO write documentation"""
    pass