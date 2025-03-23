# Ref: DOI - 10.1007/s11263-011-0474-7

from segmentation import __get_regression_params_on_neighbors
from utils import get_regression_line_parameters

import numpy as np
from numpy import cross
from numpy.linalg import norm


def __compute_dissimilarity_lines(line_a, line_b):
    """TODO write documentation"""

    p_a, v_a = line_a[:3], line_a[3:]
    p_b, v_b = line_b[:3], line_b[3:]
    # Cosine of the angle between the two directions (in range [0, 1])
    cos_angle = np.dot(v_a, v_b) / (norm(v_a) * norm(v_b))

    if 1-cos_angle < 1e-8:
        # The lines are (almost) perfectly parallel -> use a specific formula
        dist_points = norm(cross(v_a, p_b-p_a)) / norm(v_a)
    else:
        # The angle between the line is sufficient to apply the general formula
        # Source: https://www.toppr.com/guides/maths/three-dimensional-geometry/distance-between-skew-lines/
        dist_points = (abs(np.dot(p_b-p_a, cross(v_a, v_b))) 
                       / norm(cross(v_a, v_b)))

    return (1 + dist_points) / (0.01 + cos_angle)


def __compute_dissimilarity_matrix(models):
    """TODO write documentation"""
    n_models = models.shape[0]
    dissim_matrix = np.empty((n_models, n_models), dtype=np.float64)
    for i in range(n_models):
        for j in range(i, n_models):
            dissim = __compute_dissimilarity_lines(models[i], models[j])
            dissim_matrix[i, j] = dissim
            dissim_matrix[j, i] = dissim
    return dissim_matrix


def __compute_neighborhood_matrix(contacts: np.ndarray) -> np.ndarray:
    """TODO write documentation"""
    # Shapes (N, 3)
    points, directions = __get_regression_params_on_neighbors(contacts)
    models = np.concatenate([points, directions], axis=1)
    return 1/__compute_dissimilarity_matrix(models)


def __random_models_sampling(contacts, n_models):
    """TODO write documentation"""
    models = []
    generator = np.random.get_rng()
    # The samples to use for the models. Shape (n_models, 2, 3).
    # '2' is the number of samples needed to generate one model (i.e. one line)
    # '3' is the number of dimensions of each sample (i.e. contact)
    samples = generator.choice(contacts, (n_models, 2), replace=False, axis=0)

    for k in range(n_models):
        point, direction = get_regression_line_parameters(samples[k])
        models.append(np.concatenate([point, direction]))
    
    return np.stack(models)


def __get_lines_points_distance(
        contacts: np.ndarray,
        lines_points: np.ndarray,
        lines_directions: np.ndarray
) -> np.ndarray:
    """Returns the distance between each point and the given lines.
    
    ### Inputs:
    - contacts: the points in 3D space. Shape (N, 3).
    - lines_points: the points (0, p_y, p_z) by which each of the K lines 
    passes. Shape (K, 3) or (3,).
    - lines_directions: the direction vectors (1, v_y, v_z) of the lines.
    Shape must be identical to that of 'lines_points'.
    
    ### Output:
    - distances: the distance matrix between each contact (point) and each
    line. Shape (N,) or (N, K), depending on the shapes of 'lines_points' and
    'lines_directions'."""

    # Returns distances for one line. Output shape (N,)
    dist_1_line = lambda p, v: norm(np.cross(v, p-contacts), axis=1) / norm(v)

    if len(lines_points.shape) == 1:
        # Src: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        return dist_1_line(lines_points, lines_directions)
    
    # K lines
    distances = []
    for p, v in zip(lines_points, lines_directions):
        distances.append(dist_1_line(p, v))
    np.stack(distances, axis=-1)


def __compute_labels_and_energy(
        contacts, 
        models,
        neighborhood_matrix,
        lambda_weight):
    """TODO write documentation"""
    # TODO add outlier system with params = (cost, threshold)
    lines_points     = models[:,:3]
    lines_directions = models[:,3:]

    # Shape (N, K)
    distances = __get_lines_points_distance(
        contacts, lines_points, lines_directions)
    labels = distances.argmin(axis=1)
    # Distance between contact and its closest line
    model_cost = distances.min(axis=1).sum()

    deltas = labels[np.newaxis,:] != labels[:, np.newaxis]    # Potts model
    # Penalty for having similar contacts with different labels
    neighborhood_penalty = np.sum(deltas * neighborhood_matrix)

    tot_energy = model_cost + lambda_weight * neighborhood_penalty
    return labels, tot_energy


def __refine_models(contacts, models, labels):
    """TODO write documentation"""
    for k, _ in enumerate(models):
        inliers = contacts[labels == k]    # Shape (NB_INLIERS, 3)
        if len(inliers) <= 1:
            continue
        point, direction = get_regression_line_parameters(inliers)
        models[k] = np.concatenate([point, direction])
    return models


def __merge_models(model_a, weight_a, model_b, weight_b):
    """TODO write documentation"""
    return (model_a*weight_a + model_b*weight_b) / (weight_a + weight_b)


def __reduce_models(models, labels, min_inliers):
    """TODO write documentation"""
    # Account for:
    # - number of inliers
    # - similarity between models

    ### Merging similar models

    # Computing dissimilarity between all pairs of models (ignoring diagonal)
    n_models = models.shape[0]
    dissim_scores = __compute_dissimilarity_matrix(models)
    dissim_scores[range(n_models),range(n_models)] = dissim_scores.max()

    # Selecting a pair of models to merge, and the size of their support
    i, j = np.unravel_index(dissim_scores.argmin(), dissim_scores.shape)
    n_i = np.sum(labels==i)
    n_j = np.sum(labels==j)
    merged = __merge_models(models[i], n_i, models[j], n_j)

    # Updating the models
    models[i] = merged
    models = np.delete(models, j, axis=0)   # copy of 'models' without model j

    ### Removing unsupported models, starting from last 
    # (to avoid concurrent modification between modif and iteration)
    for k in range(n_models-1, -1, -1):
        if np.sum(labels==k) < min_inliers:
            models = np.delete(models, k, axis=0)

    return models


def segment_electrodes(
        contacts: np.ndarray,
        n_electrodes: int,
        return_models: bool=False
) -> np.ndarray:
    """TODO write documentation"""

    # TODO tweak hyperparams
    n_init_models = 5 * n_electrodes
    lambda_weight = 1
    neighborhood_matrix = __compute_neighborhood_matrix(contacts)
    min_inliers = 2
    max_init = 1000
    energy_tol = 1e-6

    # Proposing initial models through random sampling
    models = __random_models_sampling(contacts, n_init_models)

    # Assigning each contact to one model and computing the resulting energy
    energy_prev = None
    labels, energy_now = __compute_labels_and_energy(
        contacts, models, neighborhood_matrix, lambda_weight)
    
    init = 0
    # TODO could happen that init >= max_init AND models > n_electrodes, fix that
    while ((abs(energy_prev - energy_now) > energy_tol
           or len(models) > n_electrodes)
           and init < max_init):
        init += 1

        # Refining the models based on their spatial support
        models = __refine_models(contacts, models, labels)

        # Merging similar models, or deleting those with insufficient support
        if len(models) > n_electrodes:
            models = __reduce_models(models, labels, min_inliers)

        # Re-assigning each contact to one model and computing the resulting energy
        energy_prev = energy_now
        labels, energy_now = __compute_labels_and_energy(
            contacts, models, neighborhood_matrix, lambda_weight)
        
    if return_models:
        lines_points     = models[:,:3]
        lines_directions = models[:,3:]
        return labels, lines_points, lines_directions
    else:
        return labels
