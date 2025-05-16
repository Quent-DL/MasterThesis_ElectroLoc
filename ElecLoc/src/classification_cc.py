"""TODO write info"""

from utils import distance_matrix, estimate_intercontact_distance
from electrode_models import SegmentElectrodeModel, compute_sRsquared
from bfs import MultimodelFittingProblem, breadth_first_graph_search

import numpy as np
from typing import List, Tuple, TypeAlias


_Pair: TypeAlias = tuple[int, int]
_Group: TypeAlias = tuple[_Pair]

# HYPERPARAMETERS
MAX_ANGLE = 45
SCORE_THRESHOLD = 99    # percentage of variance explained by the models
                          # to reach within each CC. 
BFS_MAX_CHILDREN = 3    # max number of children per node. Select best ones.



############################
# TODO REMOVE: DEBUG ZONE

import time

DEBUG_PRINT = True
DEBUG_PLOT = False

import pyvista as pv
plotter = None

def plot_contacts(contacts: np.ndarray, is_interest=False) -> None:
    if len(contacts) == 0:
        return
    point_cloud = pv.PolyData(contacts)
    plotter.add_points(
        point_cloud, 
        point_size=7.5 * (2 if is_interest else 1),
        color = ("red" if is_interest else "blue"),
        render_points_as_spheres=True)
        
def plot_tree(contacts: np.ndarray, adjacency: np.ndarray,
              color: str=None, width_factor=1) -> None:
    for i, row in enumerate(adjacency):
        for j in np.where(row)[0]:
            line = pv.Line(contacts[i], contacts[j])
            plotter.add_mesh(
                line, color=color, line_width=width_factor*3)

# TODO REMOVE: DEBUG ZONE
##############################


##############################
# Generating and processing a tree of points and points of interest for models

def _kruskal(distances: np.ndarray[float]) -> np.ndarray[bool]:
    """Computes the Minimum Spanning Tree (MST) of the given undirected, 
    fully-connected graph with weights 'distances' using the Kruskal 
    algorithm and Union-Find.
    
    ### Input:
    - distances: the array of weights of the edges. Shape (N, N).
    It is assume to be symmetrical.
    
    ### Output:
    - edges: the boolean adjacency matrix of the tree, i.e. the
    edges that have been kept in the MST. Shape identical to 'distances'.
    """
    # Function generated using ChatGPT to save time

    N = distances.shape[0]
    
    # Step 1: Extract all edges (i < j)
    edge_list = [(i, j, distances[i, j]) 
                 for i in range(N) 
                 for j in range(i+1, N)]
    
    # Step 2: Sort edges by weight
    edge_list.sort(key=lambda x: x[2])
    
    # Step 3: Initialize Union-Find structure
    parent = np.arange(N)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    def union(u, v):
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            return False
        parent[root_v] = root_u
        return True

    # Step 4: Build MST
    edges = np.zeros_like(distances, dtype=bool)
    for u, v, weight in edge_list:
        if union(u, v):
            edges[u, v] = True
            edges[v, u] = True  # Maintain symmetry

    return edges

def _degrees(edges: np.ndarray[bool]) -> np.ndarray[int]:
    return edges.sum(axis=0)

def _compute_cosines(vecs: np.ndarray[float]) -> np.ndarray:
    """### Input:
    - vecs: array of vectors. Shape (N, 3).
    
    ### Output:
    - cosines: the matrix of cosines similarity between each pair of vectors.
    Shape (N, N)."""
    u = vecs[np.newaxis,:] * vecs[:,np.newaxis]    # Shape (N, N, 3)
    nrm = np.linalg.norm(vecs, axis=-1)    # Shape (N,)
    crossdot_u = np.sum(u, axis=-1)        # Shape (N, N)
    # The cosine similarity. Shape (N, N)
    cosines = crossdot_u / nrm[:,np.newaxis] / nrm[np.newaxis,:]
    return cosines


def _get_leaves(edges: np.ndarray[bool]) -> np.ndarray[int]:
    return np.where(_degrees(edges) == 1)[0]

def _get_T_intersection_vertices(edges: np.ndarray[bool]) -> np.ndarray[int]:
    """Returns the indices of the neighbors of the 3-way intersections."""
    T_intersections = np.where(_degrees(edges) == 3)[0]
    T_neighbors = np.where(edges[T_intersections].sum(axis=0) >= 1)[0]
    return T_neighbors

def _get_vertices_sudden_direction_change(
        edges: np.ndarray[bool],
        centroids_cc: np.ndarray[float],
        max_angle: float = 45
) -> np.ndarray[int]:
    """Returns the indices of all vertices with degree 2 and with sudden
     changes of directions."""
    degrees = _degrees(edges)
    vertices_kept = []
    
    for i in range(len(edges)):
        # Retrieving the indices of the vertices neighboing vertex 'i'. Shape (N,).
        neighb_global_indices = np.where(edges[i])[0]

        # Only vertices with degree 2 concerned
        if len(neighb_global_indices) != 2: continue

        # Computing the angle formed by the two neighboring edges around vertex 'i'.
        vecs = (centroids_cc[neighb_global_indices] - centroids_cc[i])    # Shape (N, 3).
        cosine_angle = _compute_cosines(vecs)[0,1]

        # In a pertfectly straight electrode, the cosine is close to -1
        # because both edges incident to 'i' point to opposite directions
        # (away from vertex 'i')
        cosine_max = np.cos((180-max_angle)*np.pi/180)    # negative value
        is_flat = cosine_angle < cosine_max

        # Detecting sudden changes of direction
        if not is_flat:
            vertices_kept.append(i)

        return np.array(vertices_kept, dtype=int)
    
def _points_of_interest(centroids_cc: np.ndarray[float]) -> list[int]:
    """Retrieves points that are candidates for being the end point of
    an electrode, even in a context with intersecting or close electrodes.
    
    ### Input:
    - centroids_cc: the centroids of the electrodes. Shape (N, 3).
    
    ### Output:
    - indices: the indices in 'centroids_cc' of the points that are candidates."""
    
    # Computing a tree of the connected component + removing some noise
    dist_matrix = distance_matrix(centroids_cc)    # Shape (N, N)
    edges = _kruskal(dist_matrix)                  # Shape (N, N), boolean

    # TODO debug remove
    if DEBUG_PLOT:
        global plotter
        plotter = pv.Plotter()
        plot_contacts(centroids_cc)
        plot_tree(centroids_cc, edges, "red", 0.5)

    # Processing the tree
    idx_interest = []

    idx_interest.append(_get_leaves(edges))
    idx_interest.append(_get_T_intersection_vertices(edges))
    idx_interest.append(_get_vertices_sudden_direction_change(
        edges, centroids_cc, MAX_ANGLE))
    
    # Retrieves the union of all arrays of integers in idx_interest
    indices_of_interest = list(np.unique(np.concatenate(idx_interest)))

    # TODO remove debug
    if DEBUG_PLOT:
        plot_tree(centroids_cc, edges, "blue", 1)
        plot_contacts(
            centroids_cc[np.array(indices_of_interest)], 
            is_interest=True)
        plotter.show()

    return indices_of_interest

##############################


##############################
# Evaluating the fitness of the models

def _compute_points_models_distances(
        centroids_cc: np.ndarray,
        models: List[SegmentElectrodeModel]
) -> np.ndarray:
    """Returns the distance between each point and the given models.
    
    ### Inputs:
    - centroids_cc: the points in 3D space. Shape (N, 3).
    - models: the list of K models (e.g. linear regression, ...).
    
    ### Output:
    - distances: the distance matrix between each centroid (point) and each
    model. Shape (N, K)"""

    # Returns distances for one line. Output shape (N,)
    distances = []
    for model in models:
        distances.append(model.compute_distance(centroids_cc))
    if len(distances) >= 1:
        return np.stack(distances, axis=-1)
    else:
        return np.empty((len(centroids_cc), 0), dtype=float)


def _compute_labels(
        centroids_cc: np.ndarray, 
        models: List[SegmentElectrodeModel]
) -> np.ndarray:
    """Classifies each centroid and assigns it the label of the closest model.
    
    ### Inputs:
    - centroids_cc: the 3D points to classify. Shape (N, 3).
    - models: the models to which the points are classified. Length K.
    
    ### Output:
    - labels: the model id in {0, ..., K-1} to which each centroid has been
    assigned. Shape (N,)."""
    # Shape (N, K)
    distances = _compute_points_models_distances(centroids_cc, models)
    labels = distances.argmin(axis=1)    # Shape (N,)
    return labels


def _compute_models_from_group(
        centroids_cc: np.ndarray[float], 
        group: _Group
) -> tuple[np.ndarray[int], List[SegmentElectrodeModel]]:
    models: List[SegmentElectrodeModel] = []
    # Compute models
    for pair in group:
        _vertices_k = centroids_cc[np.array(pair, dtype=int)]
        models.append(SegmentElectrodeModel(_vertices_k))
    
    # Compute labels (inliers) of each model 
    # then recompute models using inliers
    labels = _compute_labels(centroids_cc, models)
    for k, model in enumerate(models):
        _inliers_k = centroids_cc[labels == k]
        model.recompute(_inliers_k)
    
    return labels, models


def _score_sRsquared_group(
        centroids_cc: np.ndarray[float], 
        group: _Group
) -> float:
    """TODO write documentation"""
    if len(group) == 0:
        return 0
    labels, models = _compute_models_from_group(centroids_cc, group)
    return compute_sRsquared(models, centroids_cc, labels)


class InlierCounter():
    """This class counts the approximate number of inliers of a group of models
    using a cache mechanism. One instance of this class must be created for
    each new set of points, i.e. for each connected component"""

    def __init__(self,
                 centroids_cc: np.ndarray[float],
                 intercontact_dist: float):
        # A cache for the distances between the model computed from a pair, and all centroids
        self._cache_distances: dict[_Pair, np.ndarray[float]] = {}
        self._points = centroids_cc
        self._icd = intercontact_dist

    def _get_distances(self, group: _Group) -> np.ndarray[float]:
        # For cached pairs
        dist_cached = []
        # For non-cached pairs
        models: List[SegmentElectrodeModel] = []
        pairs_to_cache = []

        for pair in group:
            if pair in self._cache_distances:
                # If distances from model generated from pair 
                # are already computed, retrieve the values.
                dist_cached.append(self._cache_distances[pair])
            else:
                # Generate a new model
                _vertices_k = self._points[np.array(pair, dtype=int)]
                models.append(SegmentElectrodeModel(_vertices_k))
                pairs_to_cache.append(pair)

        # Computing distances of the non-cached models, then caching them
        # Shape (N, K_non_cached)
        distances = _compute_points_models_distances(self._points, models)
        for pair, dist in zip(pairs_to_cache, distances.T):
            self._cache_distances[pair] = dist

        # Adding distances of the cached models
        if len(dist_cached) >= 1:
            dist_cached = np.stack(dist_cached, axis=1)       # Shape (N, K_cached)
        else:
            # No models cached, Shape (N, 0)
            dist_cached = np.empty((len(self._points), 0), dtype=float)
        distances = np.concatenate([distances, dist_cached], axis=1)    # Shape (N, K)
        return distances

    def count_from_group(self, group: _Group) -> float:
        """TODO write documentation.
        Computes weighted number of inliers in model"""

        # Computing the final weights (accounting for all models, cached and
        # non-cached)
        distances = self._get_distances(group)
        distances = distances.min(axis=1)    # Shape (N,)
        weights = np.exp(- (distances / self._icd)**2 / 2)
        return weights.sum()

##############################


##############################
# Main algorithm

def _optimal_models(
        centroids_cc: np.ndarray
) -> Tuple[np.ndarray[int], List[SegmentElectrodeModel]]:
    """Computes the optimal group of models from the given set of points.
    A group of model is considered optimal if it minimizes the number of
    models used while ensuring a sufficient fit to the data.

    ### Input:
    - centroids_cc: the array of all 3D coordinates from which to compute the
    models. These points represent all the centroids found in a same connected 
    component. Shape (N, 3).

    ### Outputs:
    - labels: the classification labels of the centroids with the
    returned models. Shape (N,).
    - models: the optimal group of models. The length is undefined and
    contained in {1, ..., len(centroids_cc)//2}.
    """

    indices_of_interest = _points_of_interest(centroids_cc)

    icd = estimate_intercontact_distance(centroids_cc)
    inlier_counter = InlierCounter(centroids_cc, icd)

    if DEBUG_PRINT:
        t_start = time.perf_counter()
    
    bfs_problem = MultimodelFittingProblem(
        candidates = indices_of_interest,
        scoring_function = lambda group: _score_sRsquared_group(centroids_cc, group),
        children_value_function = inlier_counter.count_from_group,
        goal_score = SCORE_THRESHOLD,
        max_n_children = BFS_MAX_CHILDREN
    )

    best_group = breadth_first_graph_search(bfs_problem)

    if DEBUG_PRINT:
        t_stop = time.perf_counter()
        best_score = _score_sRsquared_group(centroids_cc, best_group)
        nb_evaluated = bfs_problem.get_number_group_scores_computed()
        print(f"=============[ BFS ]=============\n"
              f"Number of candidates: {len(indices_of_interest)}\n"
              f"Groups evaluated: {nb_evaluated}\n"
              f"Best score found: {best_score}\n"
              f"Time elapsed: {t_stop-t_start}")

    return _compute_models_from_group(centroids_cc, best_group)


def classify_centroids(
        centroids: np.ndarray,
        tags_dcc: np.ndarray
) -> Tuple[np.ndarray[int], List[SegmentElectrodeModel]]:
    """TODO write documentation"""
    
    all_models = []
    all_labels = -1 * np.ones((centroids.shape[0],), dtype=int)
    for dcc_id in np.unique(tags_dcc):
        # Computing models and labels within connected component
        centroids_cc = centroids[tags_dcc == dcc_id]
        labels_cc, models_cc = _optimal_models(centroids_cc)

        # Converting local labels to global labels and adding new models
        all_labels[tags_dcc == dcc_id] = labels_cc + len(all_models)
        all_models += models_cc

    return all_labels, all_models