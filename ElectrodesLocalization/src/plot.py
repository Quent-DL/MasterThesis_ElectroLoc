"""
This file is responsible for plotting the data
"""

import pyvista as pv
import random as random
import numpy as np
import matplotlib.pyplot as plt
from utils import LinearElectrodeModel
from typing import List

__COLOR_PALETTE = [
    
    (0, 0, 0),
    (0, 0, 255),
    (255, 150, 0),
    (0, 255, 255),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
    (255, 230, 180),


    (0, 0, 0),         # black
    (142, 202, 230),   # cyan
    (78, 78, 255),     # blue
    (255, 255, 0),     # yellow
    (220, 220, 220),   # white
    (251, 150, 0),     # orange
    (255, 0, 0),       # red
    (253, 230, 180),   # cream

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

def __get_color(i: int) -> tuple:
    # Choosing color, or new random color if there are many electrodes
    if i < len(__COLOR_PALETTE):
        color = __COLOR_PALETTE[i]
    else:
        # random electrode color
        color = [random.randint(0,255) for _ in range(3)]       
    return color 


def plot_contacts(
    contacts: np.ndarray, 
    plotter: pv.Plotter=None
) -> pv.Plotter:
    """TODO write documentation"""
    if plotter is None:
        plotter = pv.Plotter()

    point_cloud = pv.PolyData(contacts)
    plotter.add_points(
        point_cloud, point_size=5.0, 
        render_points_as_spheres=True)
    
    mean = contacts.mean(axis=0)
    plotter.camera.focal_point = mean

    return plotter


def plot_colored_electrodes(
    contacts: np.ndarray,
    labels: np.ndarray,
    plotter: pv.Plotter = None
) -> pv.Plotter:
    """TODO write documentation"""

    if plotter is None:
        plotter = pv.Plotter()

    # Iterate over each electrode and add its contacts to the plotter
    for k, e_id in enumerate(np.unique(labels)):
        color = __get_color(k)

        point_cloud = pv.PolyData(contacts[labels == e_id])
        plotter.add_points(
            point_cloud, color=color, point_size=8, 
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


# TODO remove debug
def plot_plane_proj_features(features, labels):
    fig, (ax0, ax1) = plt.subplots(1,2)

    for i, e_id in enumerate(np.unique(labels)):
        R, G, B = __get_color(i)
        color = R/255, G/255, B/255

        feats = features[labels == e_id]

        ax0.plot(feats[:,0], feats[:,1], 
                linestyle="", marker="o", markersize=2.5, color=color)
        ax1.plot(feats[:,2], feats[:,3],
                linestyle="", marker="o", markersize=2.5, color=color)
        			
    plt.show() 


# TODO remove debug
def plot_plane(center, direction, plotter: pv.Plotter):
    plane = pv.Plane(center, direction, i_size=150, j_size=150)
    plotter.add_mesh(plane, opacity=0.75)
    return plotter


def plot_linear_electrodes(
        models: List[LinearElectrodeModel], 
        func_world2vox,
        plotter: pv.Plotter
) -> pv.plotter:
    if plotter is None:
        plotter = pv.Plotter()

    for k, model in enumerate(models):
        color = __get_color(k)
        a = func_world2vox(model.point - 50 * model.direction)
        b = func_world2vox(model.point + 50 * model.direction)
        line = pv.Line(a, b)
        plotter.add_mesh(line, color=color, line_width=3)
    return plotter
    
