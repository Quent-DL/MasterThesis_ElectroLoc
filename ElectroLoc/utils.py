import numpy as np
from numpy import cross
from numpy.linalg import norm
from datetime import datetime
from typing import Tuple, Optional, Self
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from scipy.stats import gaussian_kde


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

    # A list of all the distances between a contact and its 2 closest neighbors.
    # Shape (2N,)
    distances_neigh = np.concatenate([dist_1, dist_2])

    # Kernel Density Estimation
    bandwith = 0.15    # [mm]
    x_dist = np.linspace(distances_neigh.min(), distances_neigh.max(), 10000)
    kde = gaussian_kde(distances_neigh, bw_method=0.1)
    y_density = kde(x_dist)
    return x_dist[y_density.argmax()]

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
