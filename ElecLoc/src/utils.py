import numpy as np
from numpy import cross
from numpy.linalg import norm
import nibabel as nib
from datetime import datetime
from typing import Literal, Tuple, Optional, Self
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix


class NibCTWrapper:
    def __init__(self, ct_path: str, ct_brainmask_path: str=None):
        # Thresholding and skull-stripping CT
        nib_ct = nib.load(ct_path)

        # The arrays of the CT and brain mask
        self.ct   = nib_ct.get_fdata()
        if ct_brainmask_path is not None:
            self.mask = nib.load(ct_brainmask_path).get_fdata().astype(bool)
        else:
            self.mask = np.ones_like(self.ct, dtype=bool)

        # Considering that voxels may not be square but rectangular
        # (this info is stored in the file's affine matrix)
        # we compute the sigma to apply to each axis' sigma for a Gaussian
        # to account for those different voxel side lengths
        self.affine = nib_ct.affine

    def get_voxel_size(self):
        return np.abs(np.diag(self.affine[:3,:3]))
    
    def get_center_world(self):
        """Returns the world coordinates of the image center"""
        return self.convert_vox_to_world(np.array(self.ct.shape)/2)

    def get_zoom(self):
        """Returns the zoom to apply to the image for isotropic distances."""
        vox_size = self.get_voxel_size()
        zoom_factor = vox_size / vox_size.min()
        return zoom_factor

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

    def save_contacts_mask(self, path: str, 
                           contacts: np.ndarray, r:float) -> np.ndarray:
        """Creates a binary mask in voxel space of all the given contacts,
        represented by small spheres, and saves it to the specified path
        as a '.nii.gz' file.
        
        The spheres are isotropic in physical space, and may thus appear
        as anisotropic in voxel space.
        
        ### Inputs:
        - path: the path where to save the resulting '.nii.gz' file.
        - contacts: the coordinates (in voxel space) where to add masked 
        spheres. Shape (N, 3).
        - r: the radius of the spheres (voxel space).
        
        ### Output:
        - mask: the generated binary mask which highlights the positions
        of the contacts. Boolean array with same shape as 'self.ct'."""

        # TODO fix bugy indices in this function

        # Shape (L, W, H)
        contacts_mask = np.zeros_like(self.ct, dtype=bool)

        def cube_around(shape: Tuple[int], 
                        center: np.ndarray, half_length: int) -> Tuple[int]:
            x_bfr = int(np.ceil(center[0] - max(0, center[0]-half_length)))
            x_aft = int(np.ceil(min(shape[0]-1, center[0]+half_length+1) - center[0]))
            y_bfr = int(np.ceil(center[1] - max(0, center[1]-half_length)))
            y_aft = int(np.ceil(min(shape[1]-1, center[1]+half_length+1) - center[1]))
            z_bfr = int(np.ceil(center[2] - max(0, center[2]-half_length)))
            z_aft = int(np.ceil(min(shape[2]-1, center[2]+half_length+1) - center[2]))
            return x_bfr, x_aft, y_bfr, y_aft, z_bfr, z_aft

        def mask_box(box_c: np.ndarray) -> None:
            """Returns the boolean sphere around voxel c into the flattened mask."""
            # Generated using ChatGPT
            boxL, boxW, boxH = (x_bfr+1+x_aft, y_bfr+1+y_aft, z_bfr+1+z_aft)
            x, y, z = np.meshgrid(
                np.arange(boxL), np.arange(boxW), np.arange(boxH), indexing='ij')
            box_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
            distances = np.linalg.norm(
                self.get_voxel_size()*box_points - box_c, axis=-1)
            mask_c = distances < r
            return mask_c.reshape(boxL, boxW, boxH)

        # Generating binary mask
        for c in contacts:  
            # Optimisation: only considering a box around c to compute the
            # distances => MUCH faster
            # Computing margin between the edges of the box and the center
            c = c.astype(int)
            x_bfr, x_aft, y_bfr, y_aft, z_bfr, z_aft = (
                cube_around(contacts_mask.shape, c, np.ceil(r)))
            center_box = np.array([x_bfr, y_bfr, z_bfr])
            # Computing mask within the box, then adding 
            contacts_mask[c[0]-x_bfr: c[0]+x_aft+1, 
                          c[1]-y_bfr: c[1]+y_aft+1, 
                          c[2]-z_bfr: c[2]+z_aft+1] |= mask_box(center_box)

        # Saving mask into Nifti file
        img = nib.nifti1.Nifti1Image(contacts_mask.astype(int), 
                                     self.affine)
        nib.save(img, path)


class PrecompWrapper:
    _VOX_KEYS = ['vox_x', 'vox_y', 'vox_z']
    _TAG_DCC_KEY = 'dcc_id'

    def __init__(self, path: Optional[str]=None):
        """TODO write documentation"""
        self.path = path

    def can_be_loaded(self) -> bool:
        """TODO write documentation"""
        if (self.path is None 
                or not os.path.exists(self.path)):
            return False
        df = pd.read_csv(self.path, comment="#", nrows=0)
        all_keys_present = True
        for key in self._VOX_KEYS:
            all_keys_present &= (key in df)
        return all_keys_present

    def load_precomputed_centroids(self) -> Tuple[np.ndarray]:
        """TODO write documentation"""
        df = pd.read_csv(self.path, comment="#")
        contacts_df = df[self._VOX_KEYS]
        contacts = contacts_df.to_numpy(dtype=float)
        dcc_id = df[self._TAG_DCC_KEY].to_numpy(dtype=int)
        return contacts, dcc_id

    def save_precomputed(
            self, 
            contacts: np.ndarray, 
            tags_dcc: np.ndarray
    ) -> Optional[pd.DataFrame]:
        """TODO write documentation"""
        if self.path is None:
            return
        df_content = {
            self._VOX_KEYS[0]: contacts[:,0],
            self._VOX_KEYS[1]: contacts[:,1],
            self._VOX_KEYS[2]: contacts[:,2],
            self._TAG_DCC_KEY  : tags_dcc
        }
        df = pd.DataFrame(df_content)
        # TODO fix bug float_format round not applied
        df.to_csv(
            self.path, 
            index=False,
            float_format=lambda f: round(f, 3))    
        return df


class ElectrodesInfo:
    _VOX_KEYS = ['vox_x', 'vox_y', 'vox_z']

    def __init__(self, path):
        """Initialize an instance from the information in the given CSV file.
        The CSV file must contain the following column names:
        - 'vox_x','vox_y','vox_z': the voxel coordinates of the
        entry points of each electrode.
        - 'nb_contacts': number of contacts on each electrode
            
        ### Input:
        - path: the path to the CSV file"""
        df = pd.read_csv(path, comment='#')

        # Number of electrodes. Int.
        self.nb_electrodes = len(df)
        # Entry points. Shape (NB_ELECTRODES, 3)
        self.entry_points = df[self._VOX_KEYS].to_numpy(dtype=float)
        # Number of contacts. Shape (NB_ELECTRODES,)
        self.nb_contacts = df['nb_contacts'].to_numpy(dtype=int)


class PipelineOutput(pd.DataFrame):
    _VOX_KEYS = ['vox_x', 'vox_y', 'vox_z']
    _WORLD_KEYS = ['world_x', 'world_y', 'world_z']
    _LABEL_KEY = 'electrode_id'
    _CID_KEY = 'c_id'
    # Only for ground truths
    _TAG_DCC_KEY = 'tag_dcc'

    def __init__(self,
                 vox_coords: Optional[np.ndarray] = None,
                 world_coords: Optional[np.ndarray] = None,
                 labels: Optional[np.ndarray] = None,
                 positional_ids: Optional[np.ndarray] = None,
                 ):
        super().__init__()

        if vox_coords is not None:
            self.set_vox_coordinates(vox_coords)
        if world_coords is not None:
            self.set_world_coordinates(world_coords)
        if labels is not None:
            self.set_labels(labels)
        if positional_ids is not None:
            self.set_positional_ids(positional_ids)

    @staticmethod
    def from_csv(path: str) -> Self:
        """TODO write documentation"""
        all_valid_keys = (
            PipelineOutput._VOX_KEYS 
            + PipelineOutput._WORLD_KEYS 
            + [PipelineOutput._LABEL_KEY]
            + [PipelineOutput._CID_KEY]
            + [PipelineOutput._TAG_DCC_KEY])

        # New instance with a copy of the relevant content
        instance = PipelineOutput()

        df = pd.read_csv(path, comment='#')
        for key in df.keys():
            if key in all_valid_keys:
                instance[key] = df[key]
        
        return instance
            
    def set_vox_coordinates(self, coords: np.ndarray) -> None:
        """TODO write documentation"""
        for key, values in zip(self._VOX_KEYS, coords.T):
            self[key] = values

    def get_vox_coordinates(self) -> np.ndarray:
        """TODO write documentation"""
        return self[self._VOX_KEYS].to_numpy(dtype=float)

    def set_world_coordinates(self, coords: np.ndarray) -> None:
        """TODO write documentation"""
        for key, values in zip(self._WORLD_KEYS, coords.T):
            self[key] = values

    def get_world_coordinates(self) -> np.ndarray:
        """TODO write documentation"""
        return self[self._WORLD_KEYS].to_numpy(dtype=float)

    def set_labels(self, labels: np.ndarray) -> None:
        """TODO write documentation"""
        # Checking that there are ways to identify the contacts
        _nb_vox, _nb_world = 0, 0 
        for key in self._VOX_KEYS:
            if key in self:
                _nb_vox += 1
        for key in self._WORLD_KEYS:
            if key in self:
                _nb_world += 1

        if _nb_vox == 3 or _nb_world == 3:
                self[self._LABEL_KEY] = labels
                self.sort_values(
                    by=[self._LABEL_KEY], 
                    axis='index', inplace=True)
        else: 
            raise RuntimeError("Labels cannot be set before any type of coordinates.")

    def get_labels(self) -> np.ndarray:
        """TODO write documentation"""
        dtype = self[self._LABEL_KEY].dtype
        return self[self._LABEL_KEY].to_numpy(dtype)
    
    def set_positional_ids(self, ids: np.ndarray) -> None:
        """TODO write documentation"""
        if self._LABEL_KEY in self:
            self[self._CID_KEY] = ids
            self.sort_values(
                by=[self._LABEL_KEY, self._CID_KEY], 
                axis='index', inplace=True)
        else:
            raise RuntimeError("Positional ids cannot be set before electrode ids (labels).")

    def get_positional_ids(self) -> np.ndarray:
        """TODO write documentation"""
        return self[self._CID_KEY].to_numpy(dtype=int)
    

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
        points: np.ndarray,
        samples_weights: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a linear regression using the given 3-dimensional points.
    
    ### Input:
    - points: an array of shape (K, 3) that contains the 3D coordinates of
    the points on which to perform the regression.
    - samples_weights: the weight given to each point when computing the
    regression. Shape (K,). Does not have to sum to 1. By default, all
    points are given equal weights.
    
    ### Outputs:
    - point: an array of shape (3,) that represents the point (0, p_y, p_z) by
    which the regression line passes.
    - direction: an array of shape (3,) that represents the direction vector
    (1, v_y, v_z) of the regression line.

    Assembling the two inputs, the line obtained by linear regression of 
    'points' is the set of coordinates such that 
    (x, y, z) = point + t*direction, for all real values t.
    """
    if samples_weights is None:
        samples_weights = np.ones((points.shape[0],), dtype=float)
    samples_weights /= samples_weights.sum()

    #neigh, _ = __get_vector_K_nearest(contacts, k)
    #data = np.concatenate([neigh, contacts[np.newaxis,:]])
    model = LinearRegression(fit_intercept=True)
    model.fit(points[:,:1], points[:,1:], samples_weights)
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


def estimate_intercontact_distance(
        contacts: np.ndarray
) -> Tuple[float, float]:
    """Returns an estimate of the inter-contact distance based on an histogram.
    This function computes the distance matrix of the contacts, then computes
    an estimate of the average smallest distance between two contacts 
    (ignoring outliers with unnaturally small distances).
    
    ### Input:
    - contacts: an array of shape (N, 3) that contains the 3D coordinates of
    all the contacts detected in the CT.
    
    ### Outputs:
    - dist: an estimate of the inter-contact distance.
    - dist_std: the standard deviation of the distance between each contact
    and its closest neighbor."""
    # Distance matrix of the contacts. Shape (N, N)
    distance_map = distance_matrix(contacts)
    # Ensuring that the closest detected neighbor of a contact isn't itself.
    distance_map[distance_map==0] = distance_map.max()

    # For each contact, the index of its closest neighbor + distance. Shape (N,)
    neigh_1 = distance_map.argmin(axis=1)
    dist_1  = distance_map[range(len(distance_map)), neigh_1]

    # For each contact, the distance to its 2nd closest neighbor. Shape (N,).
    # -> invalidating 1st closest neighbor
    distance_map[range(len(distance_map)), neigh_1] = distance_map.max()
    dist_2 = distance_map.min(axis=1)

    # A list of all the distances between a contact and its closest neighbors.
    # Shape (2N,)
    distances_neigh = np.concatenate([dist_1, dist_2])

    # TODO don't only take closest pairs (bias towards small intercontact distance)
    # ==> Also use 2nd closest

    # Identifying the mode of the histogram
    step = 0.2
    bins = np.arange(0, distances_neigh.max()+step, step)
    hist, _ = np.histogram(distances_neigh, bins)
    mode = hist.argmax()    # index of the modal bin

    # Applying mean to modal bin and neighboring bins (IF not on border)
    bin_min = max(0, mode-1)
    bin_max = min(len(bins)-1, mode+2)
    dist = distances_neigh[
        (bins[bin_min] < distances_neigh) 
        & (distances_neigh < bins[bin_max])].mean()

    return dist


def stable_marriage(preferences: np.ndarray, maximize=True) -> np.ndarray[int]:
    """Solves the stable marriage problem for a square matrix.
    
    ### Inputs:
    - preferences: the square matrix of pair-wise scores. Shape (N, N).
    - maximize: whether new pairs are formed by maximizing their preference
    (True) or minimizing it (False).
    
    ### Output:
    - row_to_col: the output mappings, such that 
    row_to_col[row] = column if (row, column) forms a matching pair.
    - col_to_row: the same as mappings_col_to_row but the other way
    around; col_to_row[column].
    
    Note: if (R, C) is a matching pair,
    then row_to_col[col_to_row[C]] = C
    and col_to_row[row_to_col[R]] = R."""
    assert preferences.ndim == 2, (
        "Preferences for stable marriage must be a 2D matrix.\n"
        f"Got shape {preferences.shape}.")
    assert preferences.shape[0] == preferences.shape[1], (
        "Full stable marriage requires a square matrix of preferences.\n"
        f"Got shape {preferences.shape}.")
    
    n = len(preferences)    # number of pairs to make, and size of 'preferences'
    
    unpickable = (preferences.min()-1 if maximize
                  else preferences.max()+1)
    arg_optimize = (preferences.argmax if maximize
                            else preferences.argmin)
    
    row_to_col = - np.ones((n,), dtype=int)
    col_to_row = - np.ones((n,), dtype=int)

    # Stable matching: select the best matching pair of labels,
    # then make them unpickable for future selections
    for _ in range(n):
        # Selecting best macthing pair of labels
        row, col = np.unravel_index(arg_optimize() ,(n, n))
        # Making them unpickable
        preferences[row,:] = unpickable
        preferences[:,col] = unpickable
        # Storing the pair (the actual values, not indices)
        row_to_col[row] = col
        col_to_row[col] = row

    assert -1 not in row_to_col and -1 not in col_to_row, "Bug in algorithm"
    return row_to_col, col_to_row
    

def match_and_swap_labels(labels_reference: np.ndarray, 
                 labels_pred: np.ndarray) -> np.ndarray:
    """Applies stable matching algorithm to align two arrays of labels.
    
    ### Inputs:
    - labels_reference: the labels used as reference. Integer array with
    shape (N,). Must contain number in {0, ..., X-1}.
    - labels_pred: the labels to match. Integer array with shape (N,). 
    Must contain number in {0, ..., X-1}.

    ### Output:
    - labels_matched: a copy of 'labels_pred' where the label ids have been 
    permuted to fit those in 'labels_reference'. Integer array with shape (N,).
    """
    assert labels_reference.shape == labels_pred.shape
    assert labels_reference.ndim == 1

    # An array such that labels_ids[local_idx] = label_value
    # In the confusion matrix (local_idx refers to the associated row within 
    # the confusion matrix, whereas label refers to the label's actual id/value)
    idx_to_val = np.union1d(labels_reference, labels_pred)

    # Computing the confusion matrix of the labels
    confusion = confusion_matrix(labels_reference, labels_pred)

    # local indices in the confusion matrix. Shape (Y,) with Y <= X.
    ref_to_pred_mappings, _ = stable_marriage(confusion, maximize=True)

    # Reassigning the labels
    labels_matched = np.empty_like(labels_pred, dtype=int)
    for local_ref, local_pred in enumerate(ref_to_pred_mappings):
        global_ref = idx_to_val[local_ref]
        global_pred = idx_to_val[local_pred]
        labels_matched[labels_pred == global_pred] = global_ref
    
    return labels_matched
