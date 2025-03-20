import nibabel as nib
import numpy as np
from typing import Tuple, List
from datetime import datetime
from multipledispatch import dispatch

class RawCT:
    @dispatch(str, str)
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
        self.voxel_size = np.abs(np.diag(nib_ct.affine[:3,:3] / nib_ct.affine[0,0]))

    @dispatch(np.ndarray, np.ndarray, np.ndarray)
    def __init__(self, ct: np.ndarray, mask: np.ndarray, voxel_size: np.ndarray):
        self.ct = ct.copy()
        self.mask = mask.copy()
        self.voxel_size = voxel_size.copy()

    
    def copy(self):
        """Returns a copy of the object"""
        return RawCT(self.ct, self.mask, self.voxel_size)


class Electrode:
    def __init__(self, name, head: tuple, tail: tuple):
        assert len(head) == 3 and len(tail) == 3, f"Both 'head' and 'tail' \
        must be tuples of length 3 to represent the format (x, y, z).\
        Got {len(head)} and {len(tail)}"

        self.name = name
        self.head = np.array(head, dtype=np.float32)
        self.tail = np.array(tail, dtype=np.float32)
        self.contacts = np.zeros((0,3), dtype=np.float32)

    def add_contact(self, coords: tuple) -> None:
        # Assert that the coordinates are of shape (x, y, z)
        assert len(coords) == 3

        # The [ ] around 'coords' are there to create a 2D-array, 
        # compatible with the 2D-array self.contacts
        self.contacts = np.concatenate([self.contacts, [coords]], axis=0)

    def estimate_next_contact_position(self) -> float:
        # History of last contact vectors
        history = self.contacts[1:] - self.contacts[:-1]

        # Computing the norm of the next contact vector
        # = mean of all previous distances
        distances = np.linalg.norm(history, axis=1)
        norm = distances.mean()

        # Computing the direction of the next contact vector
        # = weighted average of the last 3 contacts (in case of electrode curving)
        prev = history[-3:]
        n = prev.shape[0]    # number of previous vectors available
        # the weights applied on the last three vectors
        weights = np.array([0.225, 0.325, 0.45])   # arbitrarily chosen
        # truncating 'weights' if n is less than 3 + re-normalizing
        weights = weights[-n:] / np.linalg.norm(weights[-n:])
        weighted_prev = prev * weights[:, np.newaxis]
        dir = weighted_prev.mean(axis=0)
        unit_dir = dir / np.linalg.norm(dir)

        # Estimating the next contact's coordinates
        return self.contacts[-1] + norm * unit_dir

def get_inputs(
        ct_path: str, 
        ct_brainmask_path: str, 
        electrodes_info_path: str
    ) -> Tuple[RawCT, List[Electrode]]:
    # Reading the electrodes info from file
    electrodes = []
    with open(electrodes_info_path, "r") as file:
        for line in file.readlines():
            name, head, tail = line.strip().split("-")

            xh, yh, zh = head.strip().split(",")
            head = (float(xh), float(yh), float(zh))

            xt, yt, zt = tail.strip().split(",")
            tail = (float(xt), float(yt), float(zt))

            # Creating a new electrode
            e = Electrode(name, head, tail)
            electrodes.append(e)

    raw_ct = RawCT(ct_path, ct_brainmask_path)

    return raw_ct, electrodes

def get_structuring_element(type='cross'):
    if type == 'cube':
        return np.ones((3,3,3))
    elif type == 'cross':
        return np.array([
            [
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ],
            [
                [0,1,0],
                [1,1,1],
                [0,1,0]
            ],
            [
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ]
        ])
    else:
        raise ValueError(f"Structuring element type must be 'cube' or 'cross'. \
                         Got: {type}")


def log(msg, erase=False):
    end = "\r" if erase else None
    start = " --" if erase else ""
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    print(f"{timestamp}{start} {msg}", end=end)