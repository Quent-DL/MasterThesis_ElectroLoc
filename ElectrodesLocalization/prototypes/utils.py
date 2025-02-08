import nibabel as nib
import numpy as np
from typing import Tuple, List
from datetime import datetime

class RawCT:
    def __init__(self, ct_path, ct_brainmask_path):
        # Thresholding and skull-stripping CT
        nib_ct          = nib.load(ct_path)

        # The arrays of the CT and brain mask
        self.ct         = nib_ct.get_fdata()
        self.mask = nib.load(ct_brainmask_path).get_fdata().astype(bool)

        # Considering that voxels may not be square but rectangular
        # (this info is stored in the file's affine matrix)
        # we compute the sigma to apply to each axis' sigma for a Gaussian
        # to account for those different voxel side lengths
        self.sigmas = 1 / np.abs(np.diag(nib_ct.affine[:3,:3] / nib_ct.affine[0,0]))



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

    def mean_distance(self) -> float:
        distances = np.linalg.norm(self.contacts[1:] - self.contacts[:-1], axis=1)
        return distances.mean()
    

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




def log(msg, erase=False):
    end = "\r" if erase else None
    print(datetime.now().strftime("[%H:%M:%S]"), msg, end=end)