# Complete pipeline
from numpy.linalg import inv
from dipy.align.imaffine import AffineMap
from dipy.align.imwarp import DiffeomorphicMap
import pyvista as pv
import numpy as np

def to_homogeneous_coords(coords):
    """TODO write documentation"""
    if len(coords.shape) != 2:
        raise ValueError(f"Input should be of shape (N, 3) or (N, 4) but \
                        {coords.shape} was given")

    if coords.shape[1] == 4:
            return coords
    elif coords.shape[1] == 3:
            ones = np.ones((coords.shape[0], 1))
            hmg_coords = np.concatenate([coords, ones], axis=1)
            return hmg_coords

    raise ValueError("Input should be of shape (N, 3) or (N, 4) but "
                    + f"{coords.shape} was given")


def transform_from_ct_to_mni(
        ct_coords:      np.ndarray,
        ct_vox2world:   np.ndarray,
        anat_vox2world: np.ndarray,
        mni_vox2world:  np.ndarray,
        ct2anat_tsf: AffineMap,
        anat2mni_tsf: DiffeomorphicMap
) -> np.ndarray:
        """TODO write documentation"""
        ct_coords = ct_coords.copy()

        ### Affine transform: CT -> Anat (T1w)

        # Homogeneous coordinates in patient's CT voxel space. Shape (4, N)
        hmg_ct_vox = to_homogeneous_coords(ct_coords).T

        # Matrix of the affine transform with MRI as static and CT as moving
        # Shape (4, 4)
        ct2anat = inv(ct2anat_tsf.get_affine())
        
        # Homogeneous coordinates in patient's MRI voxel space. Shape (4, N)
        hmg_anat_vox = inv(anat_vox2world) @ ct2anat @ ct_vox2world @ hmg_ct_vox

        ### Diffeomorphic transform: MRI (Anat) -> MNI

        # Inhomogeneous coordinates in MNI's world space. Shape (N, 3)
        inhmg_mni_world = anat2mni_tsf.transform_points_inverse(
              hmg_anat_vox.T, 
              coord2world=anat_vox2world
        )
        
        # Homogeneous coordinates in MNI's voxel space. Shape (4, N)
        hmg_mni_vox = inv(mni_vox2world) @ to_homogeneous_coords(inhmg_mni_world).T

        # Inhomogeneous coordinates in MNI's voxel space. Shape (N, 3)
        inhmg_mni_vox = hmg_mni_vox[:-1,].T

        return inhmg_mni_vox



# TODO remove
def plot_contacts(
    contacts, 
    plotter: pv.Plotter = None
) -> pv.Plotter:
    """TODO write documentation"""

    if plotter is None:
        plotter = pv.Plotter()

    # Iterate over each electrode and add its contacts to the plotter
    contacts_cloud = pv.PolyData(contacts)
    plotter.add_points(contacts_cloud, point_size=10.0, 
                       render_points_as_spheres=True)
    
    # Centers the camera around the center of electrodes
    mean = contacts.mean(axis=0)
    plotter.camera.focal_point = mean

    return plotter


# TODO remove
def plot_volume(
        img: np.ndarray,
        plotter: pv.Plotter = None
) -> pv.Plotter:
    """TODO write documentation"""

    if plotter is None:
        plotter = pv.Plotter()

    grid = pv.ImageData()
    grid.dimensions = np.array(img.shape) + 1
    grid.cell_data['values'] = img.flatten(order='F')
    plotter.add_volume(grid, cmap="gray", opacity=[0,0.045/2.5])

    return plotter
