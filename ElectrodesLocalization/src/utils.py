import numpy as np
import nibabel as nib
from datetime import datetime
from typing import Literal
import pandas as pd
import os


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

    def apply_affine(
            self,
            coords: np.ndarray, 
            mode: Literal['forward', 'inverse']
    ) -> np.ndarray:
        """TODO write documentation
        
        coords: shape (3,) or (N,3)"""
        assert mode in ['forward', 'inverse']

        # The homogenous transform matrix, shape (4, 4)
        A = self.affine if mode=="forward" else np.linalg.inv(self.affine)

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
        df = pd.read_csv(self.raw_contacts_path)
        # TODO concatenate
        a = ('ct_vox_x' in df) 
        b = ('ct_vox_y' in df)
        c = ('ct_vox_z' in df)
        z = a and b and c
        return z

    def load_raw_contacts(self) -> np.ndarray:
        """TODO write documentation"""
        df = pd.read_csv(self.raw_contacts_path)
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