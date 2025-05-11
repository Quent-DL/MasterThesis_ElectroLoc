"""This file contains classes for modelling an electrode. 

ElectrodeModel is the interface with functions that all inheriting classes
should implement.
"""

from utils import get_regression_line_parameters, distance_matrix

import numpy as np
from numpy import cross, sqrt, log
from numpy.linalg import norm
from typing import Literal, Union, Self, Tuple
from overrides import override
from scipy.optimize import minimize_scalar
from abc import ABC, abstractmethod


class ElectrodeModel(ABC):
    """A curve model interface for describing generic a generic electrode."""

    # The minimum number of samples needed to compute the model parameters
    MIN_SAMPLES = 0

    @abstractmethod
    def __init__(self, samples: np.ndarray, weights: np.ndarray=None):
        """Creates a regression model for an electrode.
        
        ### Input:
        - samples: the coordinates of the K samples used to perform regression 
        and compute the model parameters. Shape (K, 3).
        - weights: the weight given to each point when computing the
        regression. Shape (K,). Does not have to sum to 1. By default, all
        points are given equal weights.
        
        ### Throws:
        - ValueError if K is insufficient to compute the number of parameters."""
        raise RuntimeError(
            "ElectrodeModel is an interface and must not be instantiated.")
    
    # TODO remove if useless
    @abstractmethod
    def compute_dissimilarity(self, other: Self) -> float:
        """Computes the dissimilarity between this model and the given one.
        
        ### Input:
        - other_model: the other model.
        
        ### Output:
        - dissimilarity: a positive measure of dissimilarity between the two 
        models. A high value means that the models are very different from one 
        another, while a value equal to zero means they are identical."""
        pass

    @abstractmethod
    def compute_distance(self, points: np.ndarray) -> np.ndarray:
        """Computes the euclidian distance between this model and the N
        given points.
        
        ### Input:
        - points: the 3D coordinates of the N points. Shape (N, 3).
        
        ### Output:
        - distance: the euclidian distance between each point and its
        closest projection onto the model. Shape (N,)."""
        pass

    @abstractmethod
    def recompute(self, samples: np.ndarray, weights: np.ndarray=None) -> None:
        """Recomputes and overwrites the model parameters using a new set
        of samples. Modifies the model internally.

        Warning: if the number of samples is insufficient to compute the model
        parameters, the model is not updated and this function call is ignored.
        
        ### Input:
        - samples: the set of N 3-dimensional coordinates used to compute
        the model parameters. Shape (N, 3).
        - weights: the weight given to each point when computing the
        regression. Shape (K,). Does not have to sum to 1. By default, all
        points are given equal weights.
        
        ### Output:
        - None"""
        pass

    @abstractmethod
    def merge(self, other: Self, w_self: float, w_other: float) -> None:
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
        pass
    
    @abstractmethod
    def project(self, contacts: np.ndarray) -> np.ndarray:
        """Projects a set of contacts onto the model. The projection of a
        contact is the point on the model with the smallest euclidian distance
        with that contact.

        ### Input:
        - contacts: the list of 3-dimensional points to project. Shape (3,) if
        a single point is projected, or (N, 3) if N points are projected.

        ### Output:
        - proj: the projection of the point(s) onto the model. Same shape as
        'contacts'."""
        pass

    @abstractmethod
    def get_sequence(self, nb_points: int, t0: float, inter_distance: float, 
                     gamma: Literal[-1, 1]) -> np.ndarray:
        """Generates a sequence of evenly-spaced points along the model.
        
        ### Inputs:
        - nb_points: the number of points to generate.
        - t0: the time parameter to start the sequence at. Reminder: since the
        model describes a curve, then the spatial coordinates (x, y, z) along
        the model can be parametrized as a function of a time t.
        - inter_distance: the distance between each point. The term "distance" 
        can either refer to the direct euclidian distance between two points,
        or the distance *along the model* between the points. It is left to
        the children class to decide which is implemented.
        - gamma: the direction to follow from t0 to compute the sequence.
        A positive gamma starts at t0 and goes towards positive t, whereas
        a negative gamma goes towards negative t.
        
        ### Output:
        - sequence: the sequence of 'nb_points' evenly-spaced by 'distance'
        along the model. Shape (nb_points, 3)."""
        pass

    @abstractmethod
    def get_gamma(self, a: np.ndarray, b: np.ndarray) -> int:
        """Returns whether the vector from a to b goes towards positive or
        negative values of t for this model.
        
        ### Inputs:
        - a: the origin of the vector. Shape (3,).
        - b: the destination of the vector. Shape (3,).
        
        ### Output:
        - gamma: whether the vector 'b-a' points towards positive (gamma = 1) 
        or negative (gamma = -1) values of t on the model."""
        pass

    @abstractmethod
    def get_projection_t(self, points: np.ndarray) -> float:
        """Computes the value of curve parameter t to describe the projection
        of a point onto the model. In other words, returns t such that
        project(points) = X(t), where X(t) is the curve model and
        project(points) is the projection of a onto this model. 
        
        ### Inputs
        - points: the point(s) to project and compute the equivalent t of. 
        Shape (3,) for a single point, or (N, 3) for N points.
        
        ### Output:
        - t: the parameter(s) that match the given 'points'. Parameter t is a 
        float if 'points' has shape (3,), or t is an array with shape (N,) 
        if 'points' has shape (N, 3).
        """
        pass


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
    def __init__(self, samples: np.ndarray, weights: np.ndarray=None):
        if len(samples) < LinearElectrodeModel.MIN_SAMPLES:
            raise ValueError("Expected at least 2 samples to create model. "
                             f"Got {len(samples)}.")
        self.recompute(samples, weights)

    # TODO remove if useless
    @override
    def compute_dissimilarity(self, other: Self) -> float:
        raise RuntimeError("Deprecated")
        if not isinstance(other, LinearElectrodeModel):
            raise ValueError("'other' must be of type LinearElectrodeModel."\
                             f"Got {type(other)}.")
        p_a, v_a = self.point, self.direction
        p_b, v_b = other.point, other.direction

        # Cosine of the angle between the two directions (in range [0, 1])
        cos_sim = abs(np.dot(v_a, v_b)) / (norm(v_a) * norm(v_b))

        if 1-cos_sim < 1e-8:
            # The lines are (almost) perfectly parallel -> use a specific formula
            # Source: https://www.toppr.com/guides/maths/three-dimensional-geometry/distance-between-parallel-lines/
            dist_points = norm(cross(v_a, p_b-p_a)) / norm(v_a)
        else:
            # The angle between the line is sufficient to apply the general formula
            # Source: https://www.toppr.com/guides/maths/three-dimensional-geometry/distance-between-skew-lines/
            dist_points = (abs(np.dot(p_b-p_a, cross(v_a, v_b))) 
                        / norm(cross(v_a, v_b)))

        # Second term used to give score 0 to identical models
        return (1 + dist_points) / (0.01 + cos_sim) - 1/(0.01+1)

    @override
    def compute_distance(self, points: np.ndarray) -> np.ndarray:
        p, v = self.point, self.direction
        # Src: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        return norm(np.cross(v, p-points), axis=1) / norm(v)
    
    @override
    def recompute(self, samples: np.ndarray, weights: np.ndarray=None) -> None:
        if len(samples) < self.MIN_SAMPLES:
            # Ignore if not enough samples given
            return
        
        self.point, self.direction = get_regression_line_parameters(
            samples, weights)

        # Setting the passage point close to the center of the samples
        self.point = self.project(samples.mean(axis=0))
        # Imposing unit directional vector
        self.direction /= norm(self.direction)

    @override
    def merge(self, other: Self, w_self: float, w_other: float) -> None:
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
    def project(self, contacts: np.ndarray) -> np.ndarray:
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
    def get_sequence(self, nb_points: int, t0: float, inter_distance: float, 
                     gamma: Literal[-1, 1]) -> np.ndarray:
        offsets = inter_distance * np.arange(nb_points).reshape((nb_points, 1))
        return self.point + self.direction * (t0 + np.sign(gamma)*offsets)

    @override
    def get_gamma(self, a: np.ndarray, b: np.ndarray) -> int:
        return np.sign(np.dot(b-a, self.direction))

    @override
    def get_projection_t(self, points: np.ndarray) -> float:
        # Using geometric interpretation of dot product to compute
        # relative t of projected points.
        p, v = self.point, self.direction
        return np.dot(points-p, v) / norm(v)**2 


class ParabolicElectrodeModel(ElectrodeModel):
    """A model for electrodes that follow a quadratic curve (= parabola).

    A 3D parabola is a curve with the following t-parametrized equations:
    - x(t) = a_x * t^2 + b_x * t + c_x
    - y(t) = a_y * t^2 + b_y * t + c_y
    - z(t) = a_z * t^2 + b_z * t + c_z
    
    This class assumes that the curvature of the electrode is relatively low
    (i.e. the electrode resembles a line with just a small curve). This means
    that the coefficients a_x, a_y, a_z should be sufficiently low. This class
    is not fit for models where the curvature is strong.
    
    Available instance attributes are:
    - coefs (np.ndarray). The array of shape (3,3) with the curve coefficients
    organized such as [[a_x, b_x, z_x], [a_y, b_y, c_y], [a_z, b_z, c_z]]."""
    MIN_SAMPLES = 3

    @override
    def __init__(self, samples: np.ndarray, weights: np.ndarray=None):
        if len(samples) < LinearElectrodeModel.MIN_SAMPLES:
            raise ValueError("Expected at least 3 samples to create model. "
                             f"Got {len(samples)}.")
        self.recompute(samples, weights)

    # TODO remove if useless
    @override
    def compute_dissimilarity(self, other: Self) -> float:
        raise RuntimeError("Deprecated")
        if not isinstance(other, ParabolicElectrodeModel):
            raise ValueError("'other' must be of type ParabolaElectrodeModel. "
                             f"Got {type(other)}.")
        
        cosine_sim = lambda a, b: np.dot(a, b) / (norm(a) * norm(b))

        v1, u1, c1 = self.coefs.T
        v2, u2, c2 = other.coefs.T

        score_c = norm(c1-c2)    # range [0, inf)

        score_u = np.abs(cosine_sim(u1, u2))    # range [0, 1]

        score_v = np.exp( (1 + cosine_sim(v1, v2))**4 / 16) - 1    # range [0,e-1]
        score_v /= np.e - 1    # range [0,1]

        return score_c / (score_u + score_v + 0.01) - 1/0.01

    @override
    def compute_distance(self, points: np.ndarray) -> np.ndarray:
        projs = self.project(points)    # Shape (N, 3)
        return np.sqrt(np.sum((points-projs)**2, axis=1))
    
    @override
    def recompute(self, samples: np.ndarray, weights: np.ndarray=None) -> None:
        if weights is None:
            weights = np.ones((samples.shape[0],), dtype=float)

        if len(samples) < self.MIN_SAMPLES:
            # Ignore if not enough samples given
            return

        # Starting by approximating the electrode as a line
        line_model = LinearElectrodeModel(samples)

        # Approximating the t values (time parameter of the curve) at which
        # the samples have been measured.
        # Division by line_model.direction omitted because unit vector.
        # Shape (N,)
        t = np.dot(samples - line_model.point, line_model.direction)

        # Computing the matrix that describes the parabola formula for all
        # measured samples
        block = np.stack([t**2, t, np.ones_like(t)]).T
        A = np.zeros((3*len(t), 9))
        A[::3, 0:3] = block
        A[1::3, 3:6] = block
        A[2::3, 6:9] = block

        # At this point, A should look like
        #   [[ t0**2  t0  1  0  0  0  0  0  0 ]
        #    [ 0  0  0  t0**2  t0  1  0  0  0 ]
        #    [  0  0  0  0  0  0 t0**2  t0  1 ]
        #    [ t1**2  t1  1  0  0  0  0  0  0 ]
        #    [ 0  0  0  t1**2  t1  1  0  0  0 ]
        #    [  0  0  0  0  0  0 t1**2  t1  1 ]
        #    [           .   .   .            ]
        #    [ 0  0  0  tn**2  tn  1  0  0  0 ]
        #    [  0  0  0  0  0  0 tn**2  tn  1 ]]

        b = samples.flatten()

        # Applying weights
        vec_weights = np.repeat(weights, 3)
        A *= vec_weights[:, np.newaxis]
        b *= vec_weights

        # Regressing the parabola
        self.coefs = np.linalg.lstsq(A, b)[0].reshape((3,3))

    @override
    def merge(self, other: Self, w_self: float, w_other: float) -> None:
        if type(other) != ParabolicElectrodeModel:
            raise ValueError("'other' must be of type ParabolaElectrodeModel."\
                             f"Got {type(other)}.")
        
        # Giving equal weights if both are zero:
        if (w_self + w_other == 0):
            w_self, w_other = 1, 1
        
        # Renaming for shorter formulas
        w_a, w_b = w_self, w_other
        self.coefs     = (self.coefs*w_a + other.coefs*w_b) / (w_a + w_b)

    @override
    def project(self, contacts: np.ndarray) -> np.ndarray:
        t_proj = self.get_projection_t(contacts)
        x = self.compute_position_at_t(t_proj)

        return x

    @override
    def get_sequence(self, nb_points: int, t0: float, inter_distance: float, 
                     gamma: Literal[-1, 1]) -> np.ndarray:

        def antiderivative_arclength(
                t: Union[float|np.ndarray]) -> Union[float|np.ndarray]:
            """Returns the solution to the arclength formula, i.e.
            the antiderivative of sqrt(x'(t)**2 + y'(t)**2 + z'(t)**2).
            
            ### Input:
            - t: the time value(s) at which to compute the value of the
            antiderivative. Can be either float (one value) or array with
            shape (N,) (multiple values).
            
            ### Output:
            - value: the result of the antiderivative of the function given
            above. Same type and shape as 't'."""
            a, b, _, d, e, _, g, h, _ = self.coefs.flatten()
            # The complete formula to integrate can be re-expressed as
            # sqrt(k*t*t + m*t + n)
            k = 4*(a*a + d*d + g*g)
            m = 4*(a*b + d*e + g*h)
            n = b*b + e*e + h*h
            # Shortcuts for the formula
            short1 = (2*k*t + m)
            short2 = sqrt(k*t*t + m*t + n)

            # Solution to the antiderivative of sqrt(k*t*t + m*t + n)
            # Src: https://www.wolframalpha.com/input?i=antiderivative+of+sqrt%28kx%5E2%2Bmx%2Bn%29
            add = short1 * short2 / (4*k)
            sub = (m*m - 4*k*n) * log(2*sqrt(k)*short2 + short1) / (8*k**(3/2))
            return add - sub
        
        def get_smallest_sufficient_deltaT(min_dist: float, t0:float) -> float:
            delta_T = 1
            anti_t0 = antiderivative_arclength(t0)
            if curve_length(t0 + delta_T) < min_dist:
                # need to increase T0
                while curve_length(t0 + delta_T) < min_dist:
                    delta_T *= 2
            else:
                # need to decrease T0
                while curve_length(t0 + delta_T) > min_dist:
                    delta_T /= 2
                delta_T *= 2    # to ensure curve_length(T) >= min_dist
            
            # Guarantee: min_dist <= curve_length(t0, t0+delta_T) <= 2*min_dist
            return delta_T

        # Idea:
        # (1) compute a linspace of lots of close t-values, starting from t0
        #         and going in direction of gamma
        # (2) compute the distance along the curve between each t-value and t0
        #         using the function 'antiderivative_arclength'
        # (3) pick all the t-values at which to place a contact. To pick the
        #         k-th t-value (starting at 0), retrieve the index of the first 
        #         t-value that gives a distance greater or equal to k * 'distance',
        #         where 'distance' is the argument of the function.
        # (4) retrieve the positions at all the picked t-values

        # a function to quickly compute curve length between t and t0
        anti_t0 = antiderivative_arclength(t0)    # only computed once
        curve_length = lambda t: np.abs(antiderivative_arclength(t) - anti_t0)
        
        delta_T = get_smallest_sufficient_deltaT(nb_points*inter_distance, t0)

        # Shapes (n,)
        n = 200*nb_points    #  total number of candidates to generate
        t_pool = np.linspace(t0, t0+gamma*delta_T, n)
        dist_t = curve_length(t_pool)

        # Shape (1, nb_points)
        dist_obj = (np.arange(nb_points)*inter_distance)[:,np.newaxis]   

        # For each distance objective, compute the first index in dist_t
        # (= index in t_pool) that fulfills it. Hence, we end up with the
        # array of all t-values at which to compute the sequence's points.
        # Shapes (nb_points,)
        picked_t_idx = np.argmax(dist_t >= dist_obj, axis=1) 
        picked_t = t_pool[picked_t_idx]

        # From the chosen t-values, we can compute the points. 
        # Shape (nb_points, 3).
        return self.compute_position_at_t(picked_t)

    @override
    def get_gamma(self, a: np.ndarray, b: np.ndarray) -> int:
        # the coefficients of 't' in the parametrized formula
        u = self.coefs[:,1]    
        
        prod = np.dot(u, b-a)
        return prod / norm(prod)
    
    def compute_position_at_t(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Computes the position of the point(s) on the parabola at the given 
        value(s) for t.
        
        ### Input:
        - t: the value(s) of time parameter. Float if single point, array of
        shape (N,) for N points.
        
        ### Output:
        - positions: the position(s) of the point(s) on the parabola at the
        given time value(s). Shape (3,) if 't' is a float, and shape (N, 3) if
        't' is of shape (N,)."""
        t_proj_vec = self.get_time_vector(t)
        X_proj = self.coefs @ t_proj_vec        # Shape (3,) or (3, N)
        return X_proj.T        # Shape (3,) or (N, 3)

    @override
    def get_projection_t(self, points: np.ndarray) -> float:

        def solve_for_one(point):
            """Solves the projection problem for one sample.
            Input: one sample of shape (3,).
            Output: the solution t (float)."""
            loss = lambda t: norm(point-self.compute_position_at_t(t))
            res_opt = minimize_scalar(loss)
            if res_opt.success:
                return res_opt.x
            else:
                print("Warning: Could not compute optimal projection time.\n"
                    f"Reason: {res_opt.message}\n")
                return 0.0

        if len(points.shape) == 1:
            # 'a' has shape (3,)
            return solve_for_one(points)
        else:
            # 'a' has shape (N, 3) => treat one sample at a time
            results = []
            for sample in points:
                results.append(solve_for_one(sample))
            return np.array(results)

    def get_time_vector(self, t: Union[float, np.ndarray]) -> np.ndarray:
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


class SegmentElectrodeModel(ElectrodeModel):
    """A segment model for straight electrodes.
    
    Available instance attributes:
    - self.point (np.ndarray): a point through which the line passes It is
    computed to be the closest to the mean of the samples used to fit
    the line. Shape (3,).
    - self.direction (np.ndarray): the unit direction vector of the line. 
    Shape (3,)."""
    MIN_SAMPLES = 2

    @override
    def __init__(self, samples: np.ndarray, weights: np.ndarray=None):
        if len(samples) < LinearElectrodeModel.MIN_SAMPLES:
            raise ValueError("Expected at least 2 samples to create model."
                             f"Got {len(samples)}.")
        self.recompute(samples, weights)

    # TODO remove if useless
    @override
    def compute_dissimilarity(self, other: Self) -> float:
        raise RuntimeError("Deprecated")
        if not isinstance(other, LinearElectrodeModel):
            raise ValueError("'other' must be of type LinearElectrodeModel."\
                             f"Got {type(other)}.")
        p_a, v_a = self.point, self.direction
        p_b, v_b = other.point, other.direction

        # Cosine of the angle between the two directions (in range [0, 1])
        cos_sim = abs(np.dot(v_a, v_b)) / (norm(v_a) * norm(v_b))

        if 1-cos_sim < 1e-8:
            # The lines are (almost) perfectly parallel -> use a specific formula
            # Source: https://www.toppr.com/guides/maths/three-dimensional-geometry/distance-between-parallel-lines/
            dist_points = norm(cross(v_a, p_b-p_a)) / norm(v_a)
        else:
            # The angle between the line is sufficient to apply the general formula
            # Source: https://www.toppr.com/guides/maths/three-dimensional-geometry/distance-between-skew-lines/
            dist_points = (abs(np.dot(p_b-p_a, cross(v_a, v_b))) 
                        / norm(cross(v_a, v_b)))

        # Second term used to give score 0 to identical models
        return (1 + dist_points) / (0.01 + cos_sim) - 1/(0.01+1)

    @override
    def compute_distance(self, points: np.ndarray) -> np.ndarray:
        p, v = self.point, self.direction

        t = self.get_projection_t(points)

        x_a, x_b = self.get_segment_nodes()
        distances_a = norm(points - x_a, axis=1)
        distances_b = norm(points - x_b, axis=1)
        # Src: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        distances_model = norm(np.cross(v, p-points), axis=1) / norm(v)

        distances = np.ones((points.shape[0]), dtype=float)
        # The distance is computed differently following three cases:
        # t < t_a,    t_b < t,    and  t_a <= t <= t_b
        cond_a = t < self.t_a
        cond_b = self.t_b < t
        cond_other = np.logical_not(cond_a | cond_b)
        
        # Filling array 'distances' with the values coming from the appropriate
        # array. 
        distances[cond_a] = distances_a[cond_a] 
        distances[cond_b] = distances_b[cond_b]
        distances[cond_other] = distances_model[cond_other] 
        return distances
    
    @override
    def recompute(self, samples: np.ndarray, weights: np.ndarray=None) -> None:
        if len(samples) < self.MIN_SAMPLES:
            # Ignore if not enough samples given
            return
        
        self.point, self.direction = get_regression_line_parameters(samples)

        # Setting the passage point close to the center of the samples
        self.point = self.project(samples.mean(axis=0))
        # Imposing unit directional vector
        self.direction /= norm(self.direction)

        # Updating the start and end times of the segment
        t = self.get_projection_t(samples)
        self.t_a = t.min()
        self.t_b = t.max()

    @override
    def merge(self, other: Self, w_self: float, w_other: float) -> None:
        if type(other) != LinearElectrodeModel:
            raise ValueError("'other' must be of type LinearElectrodeModel."\
                             f"Got {type(other)}.")
        
        raise RuntimeError("TODO update")
        
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
    def project(self, contacts: np.ndarray) -> np.ndarray:
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
    def get_sequence(self, nb_points: int, t0: float, inter_distance: float, 
                     gamma: Literal[-1, 1]) -> np.ndarray:
        offsets = inter_distance * np.arange(nb_points).reshape((nb_points, 1))
        return self.point + self.direction * (t0 + np.sign(gamma)*offsets)

    # TODO see if useful (wrapper)
    def get_segment_nodes(self) -> Tuple[np.ndarray]:
        """Returns the points delimiting the segment."""
        x_a = self.get_sequence(1, self.t_a, 0, 1)
        x_b = self.get_sequence(1, self.t_b, 0, 1)
        return x_a, x_b

    @override
    def get_gamma(self, a: np.ndarray, b: np.ndarray) -> int:
        return np.sign(np.dot(b-a, self.direction))

    @override
    def get_projection_t(self, points: np.ndarray) -> float:
        # Using geometric interpretation of dot product to compute
        # relative t of projected points.
        p, v = self.point, self.direction
        return np.dot(points-p, v) / norm(v)**2 