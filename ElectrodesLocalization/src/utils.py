import numpy as np
from numpy import cross
from numpy.linalg import norm
import nibabel as nib
from datetime import datetime
from typing import Literal, Tuple, Optional
import pandas as pd
import os
from sklearn.linear_model import LinearRegression


def log(msg: str, erase: bool=False) -> None:
    """Prints a log message in the terminal, along with a timestamp.
    
    Inputs:
    - msg: the message to print
    - erase: if True, then the log is considered temporary and will be 
    overwritten by the next one. Temporary logs are prefixed by "--" and
    end with a '\\r' when shown in the terminal."""

    end = "\r" if erase else None
    start = " --" if erase else ""
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    print(f"{timestamp}{start} {msg}", end=end)


def get_regression_line_parameters(
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a linear regression using the given 3-dimensional points.
    
    ### Input:
    - points: an array of shape (K, 3) that contains the 3D coordinates of
    the points on which to perform the regression.
    
    ### Outputs:
    - point: an array of shape (3,) that represents the point (0, p_y, p_z) by
    which the regression line passes.
    - direction: an array of shape (3,) that represents the direction vector
    (1, v_y, v_z) of the regression line.

    Assembling the two inputs, the line obtained by linear regression of 
    'points' is the set of coordinates such that 
    (x, y, z) = point + t*direction, for all real values t.
    """

    #neigh, _ = __get_vector_K_nearest(contacts, k)
    #data = np.concatenate([neigh, contacts[np.newaxis,:]])
    model = LinearRegression(fit_intercept=True)
    model.fit(points[:,:1], points[:,1:])
    return (
        np.concatenate([[0], model.intercept_]),
        np.concatenate([[1], model.coef_.ravel()])
    )


def distance_matrix(a: np.ndarray, b: np.ndarray=None) -> np.ndarray:
    """Compute the distance matrix between the points of an array, or between
    the points of two arrays.
    
    ### Input:
    - a: an array of N K-dimensional points. Shape (N, K).
    - b (optional): an array of M other K-dimensional points. Shape (M, K).
    If specified, the matrix returned contains the distance between each pair
    of points (p, q) such that 'p' belongs to 'a' and 'q' belongs to 'b'.
    If None, 'b' is set to 'a' by default.
    
    ### Output:
    - distance_matrix: an array of shape (N, M) such that 
    distance_matrix[i, j] contains the euclidian distance between a[i]
    and b[j]. If a == b, then 'distance_matrix' is symmetric."""
    if b is None:
        b = a
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return distance_matrix


class NibCTWrapper:
    def __init__(self, ct_path: str, ct_brainmask_path: str):
        # Thresholding and skull-stripping CT
        nib_ct = nib.load(ct_path)

        # The arrays of the CT and brain mask
        self.ct   = nib_ct.get_fdata()
        self.mask = nib.load(ct_brainmask_path).get_fdata().astype(bool)

        # Considering that voxels may not be square but rectangular
        # (this info is stored in the file's affine matrix)
        # we compute the sigma to apply to each axis' sigma for a Gaussian
        # to account for those different voxel side lengths
        self.affine = nib_ct.affine

    def get_voxel_size(self):
        return np.abs(np.diag(self.affine[:3,:3]))

    def __apply_affine(
            self,
            coords: np.ndarray, 
            mode: Optional[Literal['forward', 'inverse']] = "forward",
            apply_translation: Optional[bool] = True
    ) -> np.ndarray:
        """TODO write documentation
        
        coords: shape (3,) or (N,3)"""
        assert mode in ['forward', 'inverse']

        # The homogenous transform matrix, shape (4, 4)
        A = self.affine if mode=="forward" else np.linalg.inv(self.affine)
        if not apply_translation:
            A[:3,3] = 0    # removing translation coefficients

        if len(coords.shape) == 2:
            # Corods of shape (N, 3)
            # Adding 1's to get homogeneous coordinates
            ones = np.ones((coords.shape[0], 1), dtype=np.float32)
            # Shape (4, N)  
            hmg_coords = np.concatenate([coords, ones], axis=1).T
            # Getting rid of the homogeneous 1's + reshaping to (N, 3)
            return (A @ hmg_coords)[:3,:].T
        else:
            # Coords of shape (3,)
            ones = np.ones((coords.shape[0], 1), dtype=np.float32)
            hmg_coords = np.append(coords, 1).reshape((4,1))   # Shape(4, 1)
            # Getting rid of homogeneous 1 + reshaping to (3,)
            return (A @ hmg_coords)[:3].reshape((3,))
        
    def convert_vox_to_world(
            self, 
            vox_coords: np.ndarray,
            apply_translation: Optional[bool] = True
    ) -> np.ndarray:
        """TODO write documentation. Input shape (3,) or (N, 3)"""
        return self.__apply_affine(vox_coords, 'forward', apply_translation)
    
    def convert_world_to_vox(
            self, 
            vox_coords: np.ndarray,
            apply_translation: Optional[bool] = True
    ) -> np.ndarray:
        """TODO write documentation. Input shape (3,) or (N, 3)"""
        return self.__apply_affine(vox_coords, 'inverse', apply_translation)


class OutputCSV:
    def __init__(self, output_path: str, raw_contacts_path: str=None, ):
        """TODO write documentation"""
        # TODO
        self.raw_contacts_path = raw_contacts_path
        self.output_path   = output_path

    def are_raw_contacts_available(self) -> bool:
        """TODO write documentation"""
        if (self.raw_contacts_path is None 
                or not os.path.exists(self.raw_contacts_path)):
            return False
        df = pd.read_csv(self.raw_contacts_path, comment="#")
        # TODO concatenate
        a = ('ct_vox_x' in df) 
        b = ('ct_vox_y' in df)
        c = ('ct_vox_z' in df)
        z = a and b and c
        return z

    def load_raw_contacts(self) -> np.ndarray:
        """TODO write documentation"""
        df = pd.read_csv(self.raw_contacts_path, comment="#")
        contacts_df = df[['ct_vox_x', 'ct_vox_y', 'ct_vox_z']]
        return contacts_df.to_numpy(dtype=np.float32)

    def save_raw_contacts(self, contacts: np.ndarray) -> pd.DataFrame:
        """TODO write documentation"""
        df_content = {
            'ct_vox_x': contacts[:,0],
            'ct_vox_y': contacts[:,1],
            'ct_vox_z': contacts[:,2],
        }
        df = pd.DataFrame(df_content)
        # TODO fix bug float_format round not applied
        df.to_csv(
            self.raw_contacts_path, 
            index=False,
            float_format=lambda f: round(f, 3))    
        return df

    def save_output(
            self, 
            contacts: np.ndarray=None,
            electrode_ids: np.ndarray=None, 
            position_ids: np.ndarray=None
    ) -> pd.DataFrame:
        """TODO write documentation
        
        update content and write to file"""
        df_content = {
            'ct_vox_x': contacts[:,0],
            'ct_vox_y': contacts[:,1],
            'ct_vox_z': contacts[:,2],
            'e_id': electrode_ids,
            'c_id': position_ids,
        }
        df = pd.DataFrame(df_content)
        df.sort_values(
            by=['e_id', 'c_id'], 
            axis='index', inplace=True)
        # TODO fix bug float_format round not applied
        df.to_csv(
            self.output_path, 
            index=False,
            float_format=lambda f: round(f, 3))
        return df


class ElectrodesInfo:
    def __init__(self, path):
        """Initialize an instance from the information in the given CSV file.
        The CSV file must contain the following column names:
        - 'ct_vox_x','ct_vox_y','ct_vox_z': the voxel coordinates of the
        entry points of each electrode.
        - 'nb_contacts': number of contacts on each electrode
            
        ### Input:
        - path: the path to the CSV file"""
        df = pd.read_csv(path, comment='#')

        # Number of electrodes. Int.
        self.nb_electrodes = len(df)
        # Entry points. Shape (NB_ELECTRODES, 3)
        coords_columns = ['ct_vox_x','ct_vox_y','ct_vox_z']
        self.entry_points = df[coords_columns].to_numpy(dtype=np.float32)
        # Number of contacts. Shape (NB_ELECTRODES,)
        self.nb_contacts = df['nb_contacts'].to_numpy(dtype=int)