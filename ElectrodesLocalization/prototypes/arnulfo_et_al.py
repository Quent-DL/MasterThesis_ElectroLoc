import os
import nibabel as nib
import numpy as np
import pyvista as pv
import random
from datetime import datetime

##### Inputs
input_dir = "D:\QTFE_local\Python\ElectrodesLocalization\sub11\in"
ct_path           = os.path.join(input_dir, "CT.nii.gz")
ct_mask_path      = os.path.join(input_dir, "CTMask.nii.gz")
entry_points_path = os.path.join(input_dir, "entry_points.txt")
#####

##### HYPERPARAMETERS
# Masking electrodes in CT scan (Houndsfield units)
ELECTRODE_THRESHOLD = 2500

# Initial size of the box used to compute the center of mass of contacts
BOX_SIZE = 7
GAUSS_STD = 1

# Nb of attempts to find c_1
NB_ATTEMPTS_C1 = 50
# Max distance (in voxels) for two contacts to be considered the same   # TODO account for different voxel sizes (z larger than x and y) 
SAME_CONTACT_THRESHOLD = 4.0
# Max number of contacts computed in an electrode (without reaching tail) before raising an exception (due to lack of convergence)
MAX_ITER = 50
#####



###
### Fetching inputs
### 

def get_inputs(ct_path, ct_mask_path, entry_points_path):
    # Reading the entry point coordinates from file
    electrodes_names = []
    electrodes_heads = []
    electrodes_tails = []
    with open(entry_points_path, "r") as file:
        for line in file.readlines():
            name, head, tail = line.strip().split("-")

            electrodes_names.append(name)

            xh, yh, zh = head.strip().split(",")
            electrodes_heads.append([float(xh), float(yh), float(zh)])

            xt, yt, zt = tail.strip().split(",")
            electrodes_tails.append([float(xt), float(yt), float(zt)])
            
    electrodes_heads = np.array(electrodes_heads, dtype=np.float32)
    electrodes_tails = np.array(electrodes_tails, dtype=np.float32)


    # Thresholding and skull-stripping CT
    nib_ct    = nib.load(ct_path)
    ct        = nib_ct.get_fdata()
    # Considering that voxels may not be square but rectangular
    # (this info is stored in the file's affine matrix)
    # we compute the sigma to apply to each axis' sigma for a Gaussian
    # to account for those different voxel side lengths
    gaussian_sigmas    = 1 / np.abs(np.diag(nib_ct.affine[:3,:3] / nib_ct.affine[0,0]))


    ct_mask = nib.load(ct_mask_path).get_fdata().astype(bool)

    # any voxel outside the brain mask or below the intensity threshold is set to 0
    ct_masked = np.where((ct > ELECTRODE_THRESHOLD) & ct_mask, ct, 0).astype(bool)

    # TODO confirm usefulness
    #struct_elem = np.ones((9,9,7))
    #ct_masked = binary_opening(ct_masked, struct_elem)

    return ct, ct_masked, electrodes_names, electrodes_heads, electrodes_tails, gaussian_sigmas



###
### Helper function
###

def gaussian_kernel(size: int, sigmas: np.array):
    # the variance of the gaussian along each axis 
    # to account for the differences in voxel side lengths
    vx, vy, vz = sigmas**2    
    assert size % 2 != 0
    ax = np.linspace(-(size // 2), size // 2, size)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    kernel = np.exp(-(x**2 / vx + y**2 / vy + z**2 / vz) / 2)
    kernel /= kernel.sum()    # normalize the gaussian
    return kernel


def closest_contact_center(ct_masked:np.array, start: np.array, box_size: int, gaussian_sigmas) -> np.array:
    # Helper function to collect the intensities in a box around a center
    def box(arr, center, size):
        left, right = size//2, (size+1)//2
        return arr[
            center[0]-left:center[0]+right, 
            center[1]-left:center[1]+right, 
            center[2]-left:center[2]+right]
    
    # Grow region until find significant point
    start = np.array(start, dtype=int)
    def closest_point(box_size=1) -> np.array:
        region = box(ct_masked, start, box_size)
        max = region.max()
        if (max > 0):
            # Significant point found
            # The 3D coordinates of the max. The "- box_size//2" is because 
            # the origin of 'region' is not 'start' => we center region around start by shifting
            idx_max = start + (np.array(np.unravel_index(region.argmax(), region.shape)) - box_size//2)
            return idx_max
        else:
            # No significant point found => increase box size
            return closest_point(box_size+2)
        
    c = closest_point()

    # Iterative center of mass until convergence
    prev_c = np.array([-1, -1, -1])    # dummy value to ensure at least one iteration
    G = gaussian_kernel(box_size, gaussian_sigmas*GAUSS_STD)
    while np.any(np.abs(c-prev_c) > 1e-6):    # iterate until convergence of c
        prev_c = c
        c = np.vectorize(round)(c)
        X, Y, Z = box(COORDS[0], c, box_size), box(COORDS[1], c, box_size), box(COORDS[2], c, box_size)
        W = G * box(ct_masked, c, box_size)    # weight = intensity smoothed by gaussian on distance
        c = np.array([(W*X).sum(), (W*Y).sum(), (W*Z).sum()]) / W.sum()

    return c

###
### Actual algorithm
###

are_same_contact = lambda ca, cb: np.linalg.norm(ca-cb) < SAME_CONTACT_THRESHOLD

def segment_electrode(ct_masked: np.array, H: np.array, T: np.array, gaussian_sigmas) -> list:
    """Returns the center of mass of all contacts of the electrode that spans
    between coordinates H (head) and T (tail).
    
    Inputs:
    - ct_masked (numpy.array): the CT image in which the contacts are segmented. Must be of shape (X, Y, Z).
    - H (numpy.array): The coordinates of the first contact of the electrode. Must be of shape (3,).
    - T (numpy.array): The coordinates of the last contact of the electrode. Must be of shape (3,).
    
    Returns:
    contacts (numpy.array): an array of shape (N, 3) of the coordinates of all N contacts
    found in the electrode. The coordinates contacts[0] (resp. contacts[-1])
    refer to the center of mass of the contact closest to H (resp. T)."""
    contacts = []

    # TODO confirm
    #distances = []

    # Finding the first contact
    c_0 = closest_contact_center(ct_masked, H, BOX_SIZE, gaussian_sigmas)
    contacts.append(c_0)

    # Estimating the position of the second contact by starting from first contact
    # and moving along the vector T-H
    for j in range(NB_ATTEMPTS_C1):
        c_1 = contacts[0] + (j/NB_ATTEMPTS_C1) * (T-H)

        c_1 = closest_contact_center(ct_masked, c_1, BOX_SIZE, gaussian_sigmas)
        if not are_same_contact(contacts[0], c_1):
            contacts.append(c_1)
            break

    # TODO confirm
    #distances.append(np.linalg.norm(c_0-c_1))
    
    # Now that c0 and c1 are known, estimate the rest of the contacts iteratively
    # by following vector c_i - c_(i-1) and finding closest contact
    c_end = closest_contact_center(ct_masked, T, BOX_SIZE, gaussian_sigmas)
    i = 0
    while not are_same_contact(contacts[-1], c_end) and i < MAX_ITER:
        # TODO confirm
        #dir_vect = (contacts[-1] - contacts[-2]) / np.linalg.norm(contacts[-1] - contacts[-2])
        #c_i = contacts[-1] + dir_vect * np.mean(distances)
        c_i = contacts[-1] + (contacts[-1] - contacts[-2])

        c_i = closest_contact_center(ct_masked, c_i, BOX_SIZE, gaussian_sigmas)
        contacts.append(c_i)

        # TODO confirm
        #distances.append(np.linalg.norm(contacts[-1] - contacts[-2]))

        i += 1
    
    if i == MAX_ITER:
        pass # TODO handle case
        #raise RuntimeError("Max number of iterations/contacts reached but tail contact has not been found")

    return np.stack(contacts)

def segment_all_electrodes(ct_masked: np.array, electrodes_heads, electrode_tails, gaussian_sigmas) -> list:
    # Segmenting one electrode at a time, for all electrodes
    all_contacts = []
    for H, T in zip(electrodes_heads, electrode_tails):
        all_contacts.append(segment_electrode(ct_masked, H, T, gaussian_sigmas))
    return all_contacts


###
### Visualise the results
###

def visualize_contacts(ct, ct_masked, all_contacts):
    plotter = pv.Plotter()

    # To avoid displaying a too-high-resolution volume, we subsample it
    # (-> reduces computation time and RAM required)
    SF = 2    # subsample factor (must be integer)
    subsampled_ct = ct[::SF,::SF,::SF]
    subsampled_ct_masked = ct_masked[::SF,::SF,::SF]
    subsampled_all_contacts = [electrode / SF for electrode in all_contacts]
    del ct, ct_masked, all_contacts    # to avoid referencing non-sampled variables by mistake

    grid = pv.ImageData()
    grid.dimensions = np.array(subsampled_ct.shape) + 1
    grid.cell_data['values'] = subsampled_ct.flatten(order='F')
    plotter.add_volume(grid, cmap="gray", opacity=[0,0.045])

    mesh_ct = pv.wrap(subsampled_ct_masked)
    mesh_ct.cell_data['intensity'] = subsampled_ct_masked[:-1, :-1, :-1].flatten(order='F')
    vol = mesh_ct.threshold(value=1, scalars='intensity')
    plotter.add_mesh(vol, cmap='Blues', scalars='intensity', opacity=0.1)

    #plotter.add_volume(subsampled_ct, cmap="gray", opacity=[0, 0.05])

    # Iterate over each electrode and add its contacts to the plotter
    for electrode in subsampled_all_contacts:
        color = [random.random() for _ in range(3)]        # random electrode color
        point_cloud = pv.PolyData(electrode)
        plotter.add_points(point_cloud, color=color, point_size=5.0, render_points_as_spheres=True)

    
    # Centers the camera around the center of electrodes
    center_electrodes = np.concatenate(subsampled_all_contacts, axis=0).mean(axis=0)
    plotter.camera.focal_point = center_electrodes

    plotter.show()



###
### Helper function to log progression
###

def log(msg):
    print(datetime.now().strftime("[%H:%M:%S]"), msg)


###
### Main script
###

if __name__ == '__main__':
    log("Loaded libraries and tools")
    ct, ct_masked, e_names, e_heads, e_tails, gsigmas = get_inputs(ct_path, ct_mask_path, entry_points_path)
    COORDS = np.indices(ct_masked.shape)
    log("Loaded and preprocessed inputs")
    all_contacts = segment_all_electrodes(ct_masked, e_heads, e_tails, gsigmas)
    log("Segmented electrodes")
    # TODO uncomment
    visualize_contacts(ct, ct_masked, all_contacts)

    # TODO debug electrode 5 (contacts too close)
    """e5 = all_contacts[5]
    print(e5)
    print(are_same_contact(e5[0], e5[1]))
    print(np.linalg.norm(e5[0]-e5[1]))"""


""" TODO
 - Put all CT-related (ct, ct_masked, gsigmas) into one object
 - Same for electrodes entry and heads
 - Fix code (angle/distance constraint)
 - Try opening/closing preprocessing to help fix the electrode 5
 """