##### \/ \/ \/ Imports
# Loading data
import os
import nibabel as nib
from utils import get_inputs, Electrode, RawCT, log

# Ultimate erosion
import numpy as np
from math import ceil
from scipy.ndimage import zoom
from skimage.morphology import binary_erosion, reconstruction
from scipy.ndimage import label, center_of_mass

# Visualization
import pyvista as pv
import random

# Logging and utils
from typing import List
##### /\ /\ /\


##### \/ \/ \/ Inputs
input_dir = "D:\QTFE_local\Python\ElectrodesLocalization\sub11\in"
ct_path              = os.path.join(input_dir, "CT.nii.gz")
ct_mask_path         = os.path.join(input_dir, "CTMask.nii.gz")
electrodes_info_path = os.path.join(input_dir, "entry_points.txt")
##### /\ /\ /\


##### \/ \/ \/ HYPERPARAMETERS
# Masking electrodes in CT scan (Houndsfield units)
ELECTRODE_THRESHOLD = 2500

# Number of pixels in a layer of ultimate erosion
LAYER_SIZE = 32

# Nb of attempts to find c_1
NB_ATTEMPTS_C1 = 50

# Max number of contacts computed in an electrode (without reaching tail) 
# before raising an exception (due to lack of convergence)
MAX_ITER = 50
##### /\ /\ /\


def binary_ultimate_erosion(image: np.ndarray, ks=3):
    # Ref: Proakis, G. John
    result = np.zeros_like(image)
    _ndims = len(image.shape)
    struct = np.ones((ks,)*_ndims, dtype=np.int32)

    while image.sum() > 0:
        eroded        = binary_erosion(image, struct)
        reconstructed = reconstruction(eroded, image, 'dilation', struct)
        # The xor is equivalent to image - opened if they only 
        # consist of 0's and 1's, knowing that opened < image
        result |= np.logical_xor(image, reconstructed)
        image = eroded
    return result, struct


def compute_contacts_centers(raw_ct) -> np.ndarray:
    # TODO: isolate the part of mask that contains the electrodes
    # (for smaller image to process)

    UF = 2    # upsampling factor
    # TODO confirm presence of sigmas
    image = zoom(raw_ct.mask, 1/raw_ct.sigmas, order=0)

    nlayers = ceil(image.shape[0] / LAYER_SIZE)
    UP_LAYER_SIZE = UF * LAYER_SIZE    # the layer size in the upsampled image

    u_prev = zoom(np.zeros((LAYER_SIZE,)+image.shape[1:], dtype=bool), UF, order=0)    # an empty layer
    u_curr, struct = binary_ultimate_erosion(
        zoom(image[:LAYER_SIZE], UF, order=0))
    u_next, _ = binary_ultimate_erosion(
        zoom(image[LAYER_SIZE:2*LAYER_SIZE], UF, order=0))

    # A set to avoid adding twice the same contact from different layers
    centers = set()

    for lyr in range(nlayers):
        log(f"-- Layer {lyr}/{nlayers}", erase=True)

        # All the centers detected in this layer
        u_curr_centers = []

        # Opti: only execute the routine if a least one contact is in layer
        if u_curr.any():

            u_stack = np.concatenate([u_prev, u_curr, u_next])
            u_stack_labels, _ = label(u_stack, struct)
            u_layer_labels = u_stack_labels[UP_LAYER_SIZE:2*UP_LAYER_SIZE]
            labels_of_interest = np.unique(u_layer_labels)[1:]    # ignoring 0
            for lbl in labels_of_interest:
                # Assumption: a label in u_layer_labels cannot span 
                # outside the range of u_stack

                # TODO upsample and use CT for center of mass instead of the mask
                u_curr_centers.append(center_of_mass(u_stack, u_stack_labels, lbl))

            # Adding the detected centers, while accounting for the layer offset
            # the '-1' is because we want the indices in the current slice to
            # fall within [0, LAYER_SIZE) but in u_stack it is contained in
            # [LAYER_SIZE, 2*LAYER_SIZE), so we manually shift them
            layer_offset = (lyr-1) * LAYER_SIZE
            for (x, y, z) in u_curr_centers:
                centers.add((x/UF + layer_offset, y/UF, z/UF))

        # Moving up one layer to start again
        u_prev = u_curr
        u_curr = u_next
        u_next, _ = binary_ultimate_erosion(
            zoom(image[(lyr+2)*LAYER_SIZE:(lyr+3)*LAYER_SIZE], UF, order=0))
        
    # TODO presence of sigmas
    return np.stack(list(centers), dtype=np.float32) * raw_ct.sigmas


def find_closest(
        target: np.ndarray, 
        coords: np.ndarray, 
        indices: set, 
        dist_func,
        update_indices: bool = True,
) -> np.ndarray:
    """Returns the coordinates in 'coords' the closest to 'target'.
    
    Inputs:
    - target: the target coordinates. Must be of shape (3,).
    - coords: all the candidate coordinates among which we want the closest to 
    'target'. Must be of shape (N, 3).
    - indices: the set of indices of the candidates in 'coords' to consider 
    (in range {0, ..., N-1}). In 'coords', only the rows present in 'indices'
    are searched. If 'update_indices' is True, at the end of the execution, 
    the best index is removed from the set.
    - update_indices: if True, the set 'indices' is modified during the execution
    to remove the best index found. If False, 'indices' is left untouched.
    
    Output:
    - best (np.ndarray): the closest point in 'target' found, of shape (3,)."""

    best_idx, best_dist = None, 1e20

    for i in indices:
        dist = dist_func(target, coords[i]) 
        if dist < best_dist:
            best_idx  = i
            best_dist = dist
    if update_indices:
        indices.remove(best_idx)
    return coords[best_idx]


def segment_electrode(
        electrode: Electrode, 
        contacts_com: np.ndarray, 
        indices: set,
        dist_func
) -> None:
    """Returns the center of mass of all contacts of the electrode that spans
    between coordinates H (head) and T (tail).
    
    Inputs:
    - raw_ct (numpy.array): the CT image in which the contacts are segmented. 
    Must be of shape (X, Y, Z).
    - H (numpy.array): The coordinates of the first contact of the electrode. 
    Must be of shape (3,).
    - T (numpy.array): The coordinates of the last contact of the electrode. 
    Must be of shape (3,).
    
    Returns:
    contacts (numpy.array): an array of shape (N, 3) of the coordinates 
    of all N contacts found in the electrode. The coordinates contacts[0] 
    (resp. contacts[-1]) refer to the center of mass of the contact 
    closest to H (resp. T)."""

    are_same_contact = lambda a, b: dist_func(a, b) < 1e-6

    # Finding the first contact
    c_0 = find_closest(electrode.head, contacts_com, indices, dist_func, update_indices=False)
    electrode.add_contact(c_0)

    # Estimating the position of the second contact by starting from first contact
    # and moving along the vector T-H
    for j in range(1, NB_ATTEMPTS_C1+1):
        c_1_approx = c_0 + (j/NB_ATTEMPTS_C1) * (electrode.tail - electrode.head)

        c_1 = find_closest(c_1_approx, contacts_com, indices, dist_func, update_indices=False)
        if not are_same_contact(c_0, c_1):
            electrode.add_contact(c_1)
            break
    
    # Now that c_0 and c_1 are known, estimate the rest of the contacts iteratively
    # by following vector c_i - c_(i-1) and finding closest contact
    c_end = find_closest(electrode.tail, contacts_com, indices, dist_func, update_indices=False)
    iter = 0
    while not are_same_contact(electrode.contacts[-1], c_end) and iter < MAX_ITER:
        c_pp, c_p = electrode.contacts[-2], electrode.contacts[-1]
        # TODO confirm: use of unit vector and mean distance
        dir_vector = (c_p - c_pp) / np.linalg.norm(c_p - c_pp)
        c_i_approx = c_p + dir_vector * electrode.mean_distance()
        c_i = find_closest(c_i_approx, contacts_com, indices, dist_func, update_indices=False) # TODO remove update_indices=False
        electrode.add_contact(c_i)

        iter += 1
    
        if iter == MAX_ITER:
            pass # TODO handle case
            #raise RuntimeError("Max number of iterations/contacts reached 
            # but tail contact has not been found")


def segment_all_electrodes(
        electrodes: List[Electrode], 
        contacts_com: np.ndarray,
        dist_func
) -> None:
    # Segmenting one electrode at a time, for all electrodes
    n_contacts = contacts_com.shape[0]
    indices = set(range(n_contacts))
    
    for e in electrodes:
        segment_electrode(e, contacts_com, indices, dist_func)


def visualize_contacts(
        raw_ct: RawCT, 
        electrodes: List[Electrode], 
        all_contacts: np.ndarray
) -> None:
    plotter = pv.Plotter()

    # To avoid displaying a too-high-resolution volume, we subsample it
    # (-> reduces computation time and RAM required)
    SF = 2    # subsample factor (must be integer)
    ct = raw_ct.ct[::SF,::SF,::SF]
    ct_electrodes = raw_ct.mask[::SF,::SF,::SF]

    grid = pv.ImageData()
    grid.dimensions = np.array(ct.shape) + 1
    grid.cell_data['values'] = ct.flatten(order='F')
    plotter.add_volume(grid, cmap="gray", opacity=[0,0.045])

    mesh_ct = pv.wrap(ct_electrodes)
    mesh_ct.cell_data['intensity'] = ct_electrodes[:-1, :-1, :-1].flatten(order='F')
    vol = mesh_ct.threshold(value=1, scalars='intensity')
    plotter.add_mesh(vol, cmap='Blues', scalars='intensity', opacity=0.1)

    #plotter.add_volume(subsampled_ct, cmap="gray", opacity=[0, 0.05])

    # Debug: switch between colored or monochrome contacts
    if True:
        # Iterate over each electrode and add its contacts to the plotter
        for e in electrodes:
            color = [random.random() for _ in range(3)]        # random electrode color
            point_cloud = pv.PolyData(e.contacts / SF)
            plotter.add_points(point_cloud, color=color, point_size=5.0, 
                            render_points_as_spheres=True)
    else:   
        # Plotting the detected contact centers
        contacts_cloud = pv.PolyData(all_contacts / SF)
        plotter.add_points(contacts_cloud, point_size=5.0, 
                        render_points_as_spheres=True)

    
    # Centers the camera around the center of electrodes
    mean = all_contacts.mean(axis=0) / SF
    plotter.camera.focal_point = mean

    plotter.show()


# TODO archive: plots the animation of the ultimate erosion
def ultimate_erosion_and_plot(raw_ct):
    # Inputs of function
    image = raw_ct.mask
    ks = 3

    # Changing variables
    results = []
    remainings = []

    # Ultimate erosion
    result = np.zeros_like(image)
    _ndims = len(image.shape)
    struct = np.ones((ks,)*_ndims, dtype=np.int32)
    remainings.append(image.copy())

    while image.sum() > 0:
        eroded        = binary_erosion(image, struct)
        reconstructed = reconstruction(eroded, image, 'dilation', struct)
        # The xor is equivalent to image - opened if they only 
        # consist of 0's and 1's, knowing that opened < image
        result |= np.logical_xor(image, reconstructed)
        results.append(result.copy())
        image = eroded
        remainings.append(image.copy())

    # Plotting
    # TODO



if __name__ == '__main__':

    # Fetching input
    log("Loading inputs")
    raw_ct, electrodes = get_inputs(ct_path, ct_mask_path, electrodes_info_path)

    # Preprocessing the mask of the electrodes based on a threshold
    raw_ct.mask &= raw_ct.ct > ELECTRODE_THRESHOLD

    # Computing the center of mass of the electrode contacts
    log("Computing contact centers")
    # TODO remove useless if
    if False:
        contacts = compute_contacts_centers(raw_ct)
        path = os.path.join(input_dir, "centers_of_mass_UF2sigma.txt")
        np.savetxt(path, contacts)
    else:
        path = os.path.join(input_dir, "centers_of_mass_2.txt")
        contacts = np.loadtxt(path, dtype=np.float32)

    # Classifying the contacts to fill the electrode objects
    log("Classifying contacts")
    dist_func = lambda a, b: np.linalg.norm((a-b)/raw_ct.sigmas)
    segment_all_electrodes(electrodes, contacts, dist_func)

    # Vizualizing the result
    log("Plotting contacts")
    visualize_contacts(raw_ct, electrodes, contacts)