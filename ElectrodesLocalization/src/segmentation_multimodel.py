# Ref: DOI - 10.1007/s11263-011-0474-7

from segmentation import __get_vector_K_nearest
from utils import ElectrodeModel, LinearElectrodeModel

import numpy as np
from numpy import cross
from numpy.linalg import norm
from typing import List, Type


def __compute_dissimilarity_matrix(models: List[ElectrodeModel]) -> np.ndarray:
    """TODO write documentation"""
    n_models = len(models)
    dissim_matrix = np.empty((n_models, n_models), dtype=np.float64)
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models[i:]):
            dissim = model_i.compute_dissimilarity(model_j)
            dissim_matrix[i, j] = dissim
            dissim_matrix[j, i] = dissim
    return dissim_matrix


def __compute_neighborhood_matrix(
        contacts: np.ndarray,
        model_cls: Type[ElectrodeModel],
        c: float=1.0
) -> np.ndarray:
    """TODO write documentation"""
    # The neighbors of each contact. Shape (k, N, 3)
    neigh, _ = __get_vector_K_nearest(contacts, model_cls.MIN_SAMPLES)

    # For each contact (index 1) the coordinates (index 2) 
    # of itself and its neighbors (index 1). Shape (k+1, N, 3)
    all_samples = np.concatenate([neigh, contacts[np.newaxis,:]])

    models = []
    n_contacts = all_samples.shape[1]
    for i in range(n_contacts):
        model = model_cls(all_samples[:,i,:])
        models.append(model)
    return np.exp(- __compute_dissimilarity_matrix(models) / c**2)


def __random_models_sampling(
        contacts: np.ndarray, 
        n_models: np.ndarray, 
        model_cls: Type[ElectrodeModel]
) -> List[ElectrodeModel]:
    """TODO write documentation"""
    generator = np.random.default_rng()
    # The samples to use for the models. Shape (n_models, 2, 3).
    # '2' is the number of samples needed to generate one model (i.e. one line)
    # '3' is the number of dimensions of each sample (i.e. contact)
    samples = generator.choice(contacts, (n_models, 2), replace=False, axis=0)

    models = []
    for k in range(n_models):
        model = model_cls(samples[k])
        models.append(model)
    
    return models


def __compute__points_models_distances(
        contacts: np.ndarray,
        models: List[ElectrodeModel]
) -> np.ndarray:
    """Returns the distance between each point and the given models.
    
    ### Inputs:
    - contacts: the points in 3D space. Shape (N, 3).
    - models: the list of K models (e.g. linear regression, ...).
    
    ### Output:
    - distances: the distance matrix between each contact (point) and each
    model. Shape (N, K)"""

    # Returns distances for one line. Output shape (N,)
    distances = []
    for model in models:
        distances.append(model.compute_distance(contacts))
    return np.stack(distances, axis=-1)


def __compute_labels_and_energy(
        contacts: np.ndarray, 
        models: List[ElectrodeModel],
        neighborhood_matrix: np.ndarray,
        lambda_weight: float
) -> float:
    """TODO write documentation"""
    # TODO add outlier system with params = (cost, threshold)

    # Shape (N, K)
    distances = __compute__points_models_distances(contacts, models)
    labels = distances.argmin(axis=1)    # Shape (N,)
    # Sum of distances between each contact and its closest model
    model_cost = distances.min(axis=1).sum()
     
    # Using Potts' model for computing regularization term
    deltas = labels[np.newaxis,:] != labels[:, np.newaxis]

    # Penalty for having similar contacts with different labels
    neighborhood_penalty = np.sum(deltas * neighborhood_matrix)

    tot_energy = model_cost + lambda_weight * neighborhood_penalty
    return labels, tot_energy


def __recompute_models(
        contacts: np.ndarray, 
        models: List[ElectrodeModel], 
        labels: np.ndarray
) -> None:
    """TODO write documentation"""
    for k, model in enumerate(models):
        inliers = contacts[labels == k]    # Shape (NB_INLIERS, 3)
        model.recompute(inliers)


def __reduce_models(
        models: List[ElectrodeModel], 
        labels: np.ndarray, 
        min_inliers: int
) -> List[ElectrodeModel]:
    """TODO write documentation"""
    # Account for:
    # - number of inliers
    # - similarity between models

    ### Merging similar models

    # Computing dissimilarity between all pairs of models (ignoring diagonal)
    n_models = len(models)
    dissim_scores = __compute_dissimilarity_matrix(models)
    dissim_scores[range(n_models),range(n_models)] = dissim_scores.max()

    # Selecting a pair of models to merge, and the size of their support
    i, j = np.unravel_index(dissim_scores.argmin(), dissim_scores.shape)
    n_i = np.sum(labels==i)
    n_j = np.sum(labels==j)

    # Merging the model
    models[i].merge(models[j], n_i, n_j)
    models = np.delete(models, j, axis=0)   # copy of 'models' without model j

    ### Removing unsupported models, starting from last 
    # (to avoid concurrent modification between modif and iteration)
    n_models = models.shape[0]
    for k in range(n_models-1, -1, -1):
        if np.sum(labels==k) < min_inliers:
            models = np.delete(models, k, axis=0)

    return models


def segment_electrodes(
        contacts: np.ndarray,
        n_electrodes: int,
) -> np.ndarray:
    """TODO write documentation"""

    # TODO tweak hyperparams
    n_init_models = 5 * n_electrodes
    lambda_weight = 1
    min_inliers = 2
    max_init = 1000
    energy_tol = 1e-6
    neighborhood_regul_c = 2.0
    model_cls = LinearElectrodeModel

    neighborhood_matrix = __compute_neighborhood_matrix(
        contacts, model_cls, neighborhood_regul_c)

    # Proposing initial models through random sampling
    models = __random_models_sampling(
        contacts, n_init_models, model_cls)

    # Assigning each contact to one model and computing the resulting energy
    energy_prev = 1e127
    labels, energy_now = __compute_labels_and_energy(
        contacts, models, neighborhood_matrix, lambda_weight)
    
    init = 0
    # TODO could happen that init >= max_init AND models > n_electrodes, fix that
    while ((abs(energy_prev - energy_now) > energy_tol
           or len(models) > n_electrodes)
           and init < max_init):
        init += 1

        # Refining the models based on their spatial support
        __recompute_models(contacts, models, labels)

        # Merging similar models, or deleting those with insufficient support
        if len(models) > n_electrodes:
            models = __reduce_models(models, labels, min_inliers)

        # Re-assigning each contact to one model and computing the resulting energy
        energy_prev = energy_now
        labels, energy_now = __compute_labels_and_energy(
            contacts, models, neighborhood_matrix, lambda_weight)
        
    return labels, models
