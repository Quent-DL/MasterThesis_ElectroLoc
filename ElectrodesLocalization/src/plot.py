"""
This file is responsible for plotting the data
"""

import pyvista as pv
import random as random
import numpy as np

__COLOR_PALETTE = [
    (0, 0, 0),         # black
    (142, 202, 230),   # cyan
    (78, 78, 255),     # blue
    (255, 255, 0),     # yellow
    (251, 150, 0),     # orange
    (120, 0, 0),       # dark pink
    (255, 0, 0),       # red
    (253, 240, 213),   # cream

    (236, 78, 32),     # flame
    (28, 58, 19),      # pakistan green,
    (255, 255, 255),   # white

    (0, 114, 178),
    #(86, 180, 233),
    (240, 228, 66),
    (213, 94, 0),
    (204, 121, 167),
    (51, 117, 56),
    (221, 221, 221),
    (0, 156, 115),
]


def plot_colored_electrodes(
    contacts: np.ndarray,
    labels: np.ndarray,
    plotter: pv.Plotter = None
) -> pv.Plotter:
    """TODO write documentation"""

    if plotter is None:
        plotter = pv.Plotter()

    # Iterate over each electrode and add its contacts to the plotter
    for i, e_id in enumerate(np.unique(labels)):
        # Choosing color, or new random color if there are many electrodes
        if i < len(__COLOR_PALETTE):
            color = __COLOR_PALETTE[i]
        else:
            # random electrode color
            color = [random.randint(0,255) for _ in range(3)]        

        point_cloud = pv.PolyData(contacts[labels == e_id])
        plotter.add_points(
            point_cloud, color=color, point_size=8.0, 
            render_points_as_spheres=True)
    
    # Centers the camera around the center of electrodes
    mean = contacts.mean(axis=0)
    plotter.camera.focal_point = mean

    return plotter


def plot_binary_electrodes(
        ct_mask: np.ndarray,
        plotter: pv.Plotter = None
) -> pv.Plotter:
    """TODO write documentation"""

    if plotter is None:
        plotter = pv.Plotter()

    # TODO Remove: synthetic data
    if ct_mask.max() <= 0:
        return plotter


    mesh_ct = pv.wrap(ct_mask)
    mesh_ct.cell_data['intensity'] = ct_mask[:-1, :-1, :-1].flatten(order='F')
    vol = mesh_ct.threshold(value=1, scalars='intensity')
    plotter.add_mesh(vol, cmap='Blues', scalars='intensity', opacity=0.075)

    return plotter

def plot_ct(
        ct: np.ndarray,
        plotter: pv.Plotter = None
) -> pv.Plotter:
    """TODO write documentation"""

    if plotter is None:
        plotter = pv.Plotter()

    # TODO Remove: synthetic data
    if ct.max() <= 0:
        return plotter

    grid = pv.ImageData()
    grid.dimensions = np.array(ct.shape) + 1
    grid.cell_data['values'] = ct.flatten(order='F')
    plotter.add_volume(grid, cmap="gray", opacity=[0,0.045/5])

    return plotter