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


class ElectrodeModel:
    """A curve model interface for describing generic a generic electrode."""

    # The minimum number of samples needed to compute the model parameters
    MIN_SAMPLES = 0

    def __init__(self, samples: np.ndarray):
        """Creates a regression model for an electrode.
        
        ### Input:
        - samples: the coordinates of the K samples used to perform regression 
        and compute the model parameters. Shape (K, 3).
        
        ### Throws:
        - ValueError if K is insufficient to compute the number of parameters."""
        raise RuntimeError(
            "ElectrodeModel is an interface and must not be instantiated.")
    
    def compute_dissimilarity(self, other) -> float:
        """Computes the dissimilarity between this model and the given one.
        
        ### Input:
        - other_model: the other model.
        
        ### Output:
        - dissimilarity: a positive measure of dissimilarity between the two 
        models. A high value means that the models are very different from one 
        another, while a value equal to zero means they are identical."""
        raise NotImplementedError("Method not implemented by child class.")

    def compute_distance(self, points: np.ndarray) -> np.ndarray:
        """Computes the euclidian distance between this model and the
        given points.
        
        ### Input:
        - points: the 3D coordinates of the N points. Shape (N, 3).
        
        ### Output:
        - distance: the euclidian distance between each point and its
        closest projection onto the model. Shape (N,)."""
        raise NotImplementedError("Method not implemented by child class.")

    def recompute(self, samples: np.ndarray) -> None:
        """Recomputes and overwrites the model parameters using a new set
        of samples. Modifies the model internally.

        Warning: if the number of samples is insufficient to compute the model
        parameters, the model is not updated and this function call is ignored.
        
        ### Input:
        - samples: the set of N 3-dimensional coordinates used to compute
        the model parameters. Shape (N, 3).
        
        ### Output:
        - None"""
        raise NotImplementedError("Method not implemented by child class.")

    def merge(self, other, w_self: float, w_other: float) -> None:
        """Merges the parameters of two models, and overwrites this model
        with the result. Modifies the model internally.
        
        ### Inputs:
        - other: the other model to merge with this one.
        - w_self: the weight given to the parameters of this model.
        - w_other: the weight given to the parameters of the other model.
        
        Note: the merged model is computed following the formula:

        params_merged = (w_self * params_self + w_other * params_other)
        / (w_self + w_other)
        
        ### Output:
        None"""
        raise NotImplementedError("Method not implemented by child class.")
    
    def project(self, contacts: np.ndarray) -> np.ndarray:
        """Projects a set of contacts onto the model. The projection of a
        contact is the point on the model with the smallest euclidian distance
        with that contact.

        ### Input:
        - contacts: the list of 3-dimensional points to project. Shape (3,) if
        a single point is projected, or (N, 3) if N points are projected.

        ### Output:
        - proj: the projection of the point(s) onto the model. Same shape as
        'contacts."""
        raise NotImplementedError("Method not implemented by child class.")

    def get_sequence(self, nb_points: int, t0: float, 
                     distance: float, gamma: Literal[-1, 1]) -> np.ndarray:
        """Generates a sequence of evenly-spaced points along the model.
        
        ### Inputs:
        - nb_points: the number of points to generate.
        - t0: the time parameter to start the sequence at. Reminder: since the
        model describes a curve, then the spatial coordinates (x, y, z) along
        the model can be parametrized as a function of a time t.
        - distance: the distance between each point. The term "distance" can
        either refer to the direct euclidian distance between two points,
        or the distance *along the model* between the points. It is left to
        the children class to decide which is implemented.
        - gamma: the direction to follow from t0 to compute the sequence.
        A positive gamma starts at t0 and goes towards positive t, whereas
        a negative gamma goes towards negative t.
        
        ### Output:
        - sequence: the sequence of 'nb_points' evenly-spaced by 'distance'
        along the model. Shape (nb_points, 3)."""
        raise NotImplementedError("Method not implemented by child class.")

    def get_gamma(self, a: np.ndarray, b: np.ndarray) -> int:
        """Returns whether the vector from a to b goes towards positive or
        negative values of t for this model.
        
        ### Inputs:
        - a: the origin of the vector. Shape (3,).
        - b: the destination of the vector. Shape (3,).
        
        ### Output:
        - gamma: whether the vecteur 'b-a' points towards positive (gamma = 1) 
        or negative (gamma = -1) values of t on the model."""
        raise NotImplementedError("Method not implemented by child class.")

    def get_projection_t(self, a: np.ndarray) -> float:
        """Computes the value of curve parameter t to describe the projection
        of a point onto the model. In other words, returns t such that
        project(a) = X(t), where X(t) is the curve model and
        project(a) is the projection of a onto this model. 
        
        ### Inputs
        - a: the point(s) to project and compute the equivalent t of. 
        Shape (3,) for a single point, or (N, 3) for N points.
        
        ### Output:
        - t: the parameter(s) that match the given a. Parameter t is a float
        if a has shape (3,), or t is an array with shape (N,) if a has shape
        (N, 3).
        """
        raise NotImplementedError("Method not implemented by child class.")


class LinearElectrodeModel(ElectrodeModel):
    """A linear model for straight electrodes"""
    MIN_SAMPLES = 2

    def __init__(self, samples):
        if len(samples) < LinearElectrodeModel.MIN_SAMPLES:
            raise ValueError("Expected at least 2 samples to create model."
                             f"Got {len(samples)}.")
        self.recompute(samples)

    def compute_dissimilarity(self, other):
        if type(other) != LinearElectrodeModel:
            raise ValueError("'other' must be of type LinearElectrodeModel."\
                             f"Got {type(other)}.")
        p_a, v_a = self.point, self.direction
        p_b, v_b = other.point, other.direction

        # Cosine of the angle between the two directions (in range [0, 1])
        cos_angle = abs(np.dot(v_a, v_b)) / (norm(v_a) * norm(v_b))

        if 1-cos_angle < 1e-8:
            # The lines are (almost) perfectly parallel -> use a specific formula
            # Source: https://www.toppr.com/guides/maths/three-dimensional-geometry/distance-between-parallel-lines/
            dist_points = norm(cross(v_a, p_b-p_a)) / norm(v_a)
        else:
            # The angle between the line is sufficient to apply the general formula
            # Source: https://www.toppr.com/guides/maths/three-dimensional-geometry/distance-between-skew-lines/
            dist_points = (abs(np.dot(p_b-p_a, cross(v_a, v_b))) 
                        / norm(cross(v_a, v_b)))

        # Second term used to give score 0 to identical models
        return (1 + dist_points) / (0.01 + cos_angle) - 1/(0.01+1)

    def compute_distance(self, points):
        p, v = self.point, self.direction
        # Src: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        return norm(np.cross(v, p-points), axis=1) / norm(v)
    
    def recompute(self, samples):
        if len(samples) < self.MIN_SAMPLES:
            # Ignore if not enough samples given
            return
        
        self.point, self.direction = get_regression_line_parameters(samples)

        # Setting the passage point close to the center of the samples
        self.point = self.project(samples.mean(axis=0))
        # Imposing unit directional vector
        self.direction /= norm(self.direction)


    def merge(self, other, w_self, w_other):
        if type(other) != LinearElectrodeModel:
            raise ValueError("'other' must be of type LinearElectrodeModel."\
                             f"Got {type(other)}.")
        
        # Giving equal weights if both are zero:
        if (w_self + w_other == 0):
            w_self, w_other = 1, 1
        
        # Renaming for shorter formulas
        p_a, v_a = self.point, self.direction
        p_b, v_b = other.point, other.direction
        w_a, w_b = w_self, w_other

        self.point     = (p_a*w_a + p_b*w_b) / (w_a + w_b)
        self.direction = (v_a*w_a + v_b*w_b) / (w_a + w_b)

    def project(self, contacts):
        # For shorter formulas
        p, v = self.point, self.direction
        if len(contacts.shape) == 1:
            # Contacts of shape (3,)
            return p + np.dot(contacts-p, v) / np.dot(v, v) * v
        else:
            # Contacts of shape (N, 3)
            dots = np.dot(contacts-p, v).reshape((len(contacts), 1))
            v_repeated = np.tile(v, (len(contacts), 1))
            return p + dots / np.dot(v, v) * v_repeated

    def get_sequence(self, nb_points, t0, distance, gamma):
        offsets = distance * np.arange(nb_points).reshape((nb_points, 1))
        return self.point + self.direction * (t0 + np.sign(gamma)*offsets)

    def get_gamma(self, a, b):
        return np.sign(np.dot(b-a, self.direction))

    def get_projection_t(self, a):
        # proj(a)[j] = self.point[j] + t * self.direction[j]
        # choose j such that self.direction[j] is not close to 0 
        # (to avoid zero division-)
        proj = self.project(a)
        j = np.argmax(np.abs(self.direction))
        if len(a.shape) == 1:
            # 'a' and 'proj' have shape (3,)
            return (proj - self.point)[j] / self.direction[j]
        else:
            # 'a' and 'proj' have shape (N, 3)
            return (proj[:,j] - self.point[j]) / self.direction[j]


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