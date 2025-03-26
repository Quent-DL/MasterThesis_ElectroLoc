"""This file does NOT belong to the pipeline. Its sole purpose is to provide
a quick script to create plots and visualizations. It must not be imported
from another module."""

# Preventing another module from importing this file
if __name__ != '__main__':
    raise RuntimeError("This module is not meant to be imported")

import pyvista as pv
from contacts_isolation import get_contacts
from segmentation import __get_vector_K_nearest, __feature_proj_plane_new
from sklearn.linear_model import LinearRegression
import numpy as np
from utils import NibCTWrapper
import os
from scipy.ndimage import zoom
from plot import plot_ct, plot_binary_electrodes


def __get_dir_regression(contacts, k):
    neigh, _ = __get_vector_K_nearest(contacts, k)
    data = np.concatenate([neigh, contacts[np.newaxis,:]])
    intercepts = []
    dirs = []
    # Fitting one regression for each set of contact and neighbors
    for i in range(contacts.shape[0]):
        x = data[:,i,:1]
        y = data[:,i,1:]
        model = LinearRegression(fit_intercept=True)
        model.fit(x, y)
        intercepts.append(model.intercept_)
        dirs.append(model.coef_.ravel())
    return np.stack(intercepts), np.stack(dirs)


def plot_contacts(contacts, plotter=None, point_size=4.0, color=None, update_camera=False) -> None:
    if plotter is None:
        plotter = pv.Plotter()


    # Plotting the detected contact centers
    contacts_cloud = pv.PolyData(contacts)
    plotter.add_points(
        contacts_cloud, point_size=point_size, color=color,
        render_points_as_spheres=True)

    # Centers the camera around the center of electrodes
    if update_camera:
        mean = contacts.mean(axis=0)
        plotter.camera.focal_point = mean

    return plotter


def plot_planes(xmin, xmax, y, z, plotter=None):
    if plotter is None:
        plotter = pv.Plotter()

    kwargs = {
        'i_size': 125,
        'j_size': 125,
        'i_resolution': 5,
        'j_resolution': 5,
    }

    plane_min = pv.Plane(center=(xmin, y, z), direction=(1,0,0), **kwargs)
    plane_max = pv.Plane(center=(xmax, y, z), direction=(1,0,0), **kwargs)

    plane_min.point_data.clear()
    plane_max.point_data.clear()

    plotter.add_mesh(plane_min, opacity=0.2)
    plotter.add_mesh(plane_max, opacity=0.2)



if __name__ == '__main__':

    contacts = get_contacts(synthetic=True)

    xmin = 250 # contacts[0].min()
    xmax = 400 # contacts[0].max()
    ymean = 150
    zmean = 150

    plotter = pv.Plotter()

    # Plotting all contacts and projection planes
    plot_contacts(contacts, plotter, 5, update_camera=True)
    plot_planes(xmin, xmax, ymean, zmean, plotter)

    neighbors, _ = __get_vector_K_nearest(contacts, 3)

    i = 45
    c_i     = np.array([contacts[i]])
    neigh_i = neighbors[:,i]
    
    # Plotting special contacts
    #plot_contacts(c_i, plotter, 15, 'blue')
    #plot_contacts(neigh_i, plotter, 15, 'orange')

    # Plotting line
    
    xmin_arr = xmin * np.ones((contacts.shape[0], 1))
    proj_min = __feature_proj_plane_new(contacts, xmin)
    proj_min = np.concatenate([xmin_arr, proj_min], axis=1)

    xmax_arr = xmax * np.ones((contacts.shape[0], 1))
    proj_max = __feature_proj_plane_new(contacts, xmax)
    proj_max = np.concatenate([xmax_arr, proj_max], axis=1)
    
    #plotter.add_lines(np.array([proj_min, proj_max]), color='grey', width=1)

    # Plotting intersections
    plot_contacts(proj_min, plotter, 8, 'orange')
    plot_contacts(proj_max, plotter, 8, 'green')

    #plotter.show()

    
    # Plotting contacts

    # Inputs
    input_dir         = "D:\\QTFE_local\\Python\\ElectrodesLocalization\\sub11\\in"
    ct_path           = os.path.join(input_dir, "CT.nii.gz")
    ct_brainmask_path = os.path.join(input_dir, "CTMask.nii.gz")
    contacts_path     = os.path.join(input_dir, "contacts.txt")

    # Loading the data
    ct_object = NibCTWrapper(ct_path, ct_brainmask_path)

    factors = np.abs(np.diag(ct_object.affine)[:-1])
    factors /= factors[0]
    ct_object.ct = zoom(ct_object.ct, factors)
    ct_object.mask = zoom(ct_object.mask, factors)
    ct_object.mask &= (ct_object.ct > 2500)

    plotter = pv.Plotter()
    plot_ct(ct_object.ct, plotter)
    #plot_binary_electrodes(ct_object.mask, plotter)
    plotter.show()