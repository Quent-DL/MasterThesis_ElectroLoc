import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
import nibabel as nib
from datetime import datetime


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


class Electrode:
    """TODO documentation"""
    
    def __init__(self, contacts: np.ndarray, ct_shape):
        """TODO write documentation"""
        ct_center = np.array(ct_shape, dtype=np.float32) / 2

        # Sorting the contact by their value on the main axis of the electrode
        pca = PCA(n_components=1)
        scores = pca.fit_transform(contacts).ravel()   # ravel to convert (N,1) to (N,)
        sorted_contacts = contacts[np.argsort(scores)]

        # If necessary, reversing the order of the contacts of the electrode so 
        # that the first contact of the array is the deepest
        # (i.e. the closest to the center of the ct)
        if norm(sorted_contacts[0]-ct_center) > norm(sorted_contacts[-1]-ct_center):
            # First contact is deeper than last => reverse the electrode
            sorted_contacts = np.flip(sorted_contacts, axis=0)

        self.contacts = sorted_contacts


class NibCTWrapper:
    def __init__(self, ct_path: str, ct_brainmask_path: str):
        # Thresholding and skull-stripping CT
        nib_ct          = nib.load(ct_path)

        # The arrays of the CT and brain mask
        self.ct         = nib_ct.get_fdata()
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
            type: str
    ) -> np.ndarray:
        # TODO write documentation
        assert type in ['forward', 'inverse']

        # The homogenous transform matrix, shape (4, 4)
        A = self.affine if type=="forward" else np.linalg.inv(self.affine)

        # Adding 1's to get homogeneous coordinates
        ones = np.ones((coords.shape[0], 1), dtype=np.float32)
        x = np.concatenate([coords, ones], axis=1).T    # Shape (4, N)

        # Shape (N, 3)
        return (A @ x)[:3,:].T    # getting rid of the homogeneous 1's
        