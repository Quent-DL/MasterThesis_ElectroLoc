"""This file contains classes for modelling an electrode. 

ElectrodeModel is the interface with functions that all inheriting classes
should implement.
"""

from utils import get_regression_line_parameters

import numpy as np
from numpy import cross
from numpy.linalg import norm
from typing import Literal, Union
from overrides import override
from scipy.optimize import minimize_scalar


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
        - gamma: whether the vector 'b-a' points towards positive (gamma = 1) 
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
        - t: the parameter(s) that match the given 'a'. Parameter t is a float
        if 'a' has shape (3,), or t is an array with shape (N,) if 'a' has 
        shape (N, 3).
        """
        raise NotImplementedError("Method not implemented by child class.")


class LinearElectrodeModel(ElectrodeModel):
    """A linear model for straight electrodes.
    
    Available instance attributes:
    - self.point (np.ndarray): a point through which the line passes It is
    computed to be the closest to the mean of the samples used to fit
    the line. Shape (3,).
    - self.direction (np.ndarray): the unit direction vector of the line. 
    Shape (3,)."""
    MIN_SAMPLES = 2

    @override
    def __init__(self, samples):
        if len(samples) < LinearElectrodeModel.MIN_SAMPLES:
            raise ValueError("Expected at least 2 samples to create model."
                             f"Got {len(samples)}.")
        self.recompute(samples)

    @override
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

    @override
    def compute_distance(self, points):
        p, v = self.point, self.direction
        # Src: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        return norm(np.cross(v, p-points), axis=1) / norm(v)
    
    @override
    def recompute(self, samples):
        if len(samples) < self.MIN_SAMPLES:
            # Ignore if not enough samples given
            return
        
        self.point, self.direction = get_regression_line_parameters(samples)

        # Setting the passage point close to the center of the samples
        self.point = self.project(samples.mean(axis=0))
        # Imposing unit directional vector
        self.direction /= norm(self.direction)

    @override
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

    @override
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

    @override
    def get_sequence(self, nb_points, t0, distance, gamma):
        offsets = distance * np.arange(nb_points).reshape((nb_points, 1))
        return self.point + self.direction * (t0 + np.sign(gamma)*offsets)

    @override
    def get_gamma(self, a, b):
        return np.sign(np.dot(b-a, self.direction))

    @override
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



        """Creates the array [t^2, t, 1] to multiply to self.coefs to compute
        the position of a point on the parabola at the specified time t.
        
        ### Inputs:
        - t: the time parameter(s) at which the point(s) are created. Can be
        either a float, to describe a single point, or an array of shape (N,)
        to describe N points.
        
        ### Output:
        - t_matrix: the array or matrix that contains the column 
        [t_i^2, t_i, 1] for each sample t_i given in the input. Shape (3,)
        if the argument t is a float, and shape (3, N) if t is an array
        with shape (N,)."""
        return np.array([t**2, t, np.ones_like(t)])