"""TODO write info"""

from utils import distance_matrix
from electrode_models import SegmentElectrodeModel

import numpy as np
from typing import List, Tuple
from itertools import combinations
import warnings


############################
# TODO REMOVE: DEBUG ZONE

DEBUG_PRINT = False
DEBUG_PLOT = False

import pyvista as pv
plotter = None

def plot_contacts(contacts: np.ndarray) -> None:
    if len(contacts) == 0:
        return
    point_cloud = pv.PolyData(contacts)
    plotter.add_points(
        point_cloud, 
        point_size=7.5,
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

SCORE_THRESHOLD = 0.98
def _compute_score(
        models: List[SegmentElectrodeModel], 
        centroids_cc: np.ndarray, 
        labels: np.ndarray
) -> float:
    """Computes the overall score of the given group of models.
    
    ### Inputs:
    - models: the models in the groupe to evaluate. Length can be arbitrary.
    - centroids_cc: the points used to compute the score of each model.
    Shape (N, 3).
    - labels: the model to which each centroid in 'centroids_cc' has been
    assigned. The expression 'labels[i]' returns 'k' if the i-th centroid
    in centroids_cc has been assigned to the model contained in 'models[k]'.
    Shape (N,).
    
    ### Output:
    - score: the score of the groupe of models."""
    sSST = 0    # the summed Total SS across all models
    sSSR = 0    # the summed Regression SS across all models
    for k, model in enumerate(models):
        inliers_k = centroids_cc[labels == k]
        center_k = np.mean(inliers_k, axis=0)
        proj_k = model.project(inliers_k)

        SST_k = np.sum(np.linalg.norm(inliers_k - center_k, axis=1)**2)
        SSR_k = np.sum(np.linalg.norm(proj_k    - center_k, axis=1)**2)

        sSST += SST_k
        sSSR += SSR_k

    s_R_squared = sSSR / sSST
    return s_R_squared


# Generated using ChatGPT to save time
def _kruskal(distances: np.ndarray) -> np.ndarray:
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

def _remove_solo_noise(edges: np.ndarray) -> None:
    """Slightly rearranges a tree such that all leaves connected to
    vertices of degree 3 are not leaves anymore.
    
    Specifically, the goal is to convert all occurences of:
                        [I]
                         |
        ... --- [K] --- [J] --- [M] --- ...
        
    where vertex I is a leaf as defined above, to:

                 ------ [I]
                 |       |
        ... --- [K]     [J] --- [M] --- ...

    where I is not a leaf anymore.
    
    ### Input:
    - edges: the adjacency matrix of the tree. Shape (N, N). 
    Assumed to be boolean.

    ### Output:
    - The function returns nothing, but 'edges' is modified in place.
    """
    degree = lambda idx: edges[idx].sum()
    for i, neighs_i in enumerate(edges):
        if degree(i) == 1:    # If i is a leaf (only one neighbor)
            j = neighs_i.argmax()    # The int neighbor of i
            if degree(j) == 3:
                neighs_j = np.where(edges[j])[0]
                # The other two neighbors of j that are not i
                k, m = set(list(neighs_j)).difference([i])
                # Replacing edge (j, k) by (i, k)
                # so that i is not a leaf anymore (linked to j and k)
                edges[j,k] = False; edges[k,j] = False
                edges[i,k] = True;  edges[k, i] = True

def _remove_big_angles(edges: np.ndarray, centroids_cc: np.ndarray,
                       max_angle: float) -> np.ndarray:
    """TODO write documentation.
    Removes edges based on alignment.
    An edge E is removed from edges if all other edges touching
    any of the two incident nodes form an angle with E outside the range 
    [180-max_angle, 180].
    'max_angle' expressed in edgrees."""
    copy_edges = np.zeros_like(edges)
    # centroid.shape is (3,). neighb_edges.shape is (N,) 
    for i in range(len(edges)):
        neighb_edges = edges[i]
        neighb_global_indices = np.where(neighb_edges)[0]

        # This algorithm only applies to vertices with
        # at least two neighbors
        if len(neighb_global_indices) < 2:
            copy_edges[neighb_global_indices, i] = 1
            continue

        # The vectors from vertex 'i' to its neighbors
        vecs = (centroids_cc[neighb_global_indices] - centroids_cc[i])    # Shape (Nb, 3)

        # Computing the cosine similarity between all pairs of edges
        # among the 'Nb' edges that are incident to the vertex 'i'.
        u = vecs[np.newaxis,:] * vecs[:,np.newaxis]    # Shape (Nb, Nb, 3)
        nrm = np.linalg.norm(vecs, axis=-1)    # Shape (Nb,)
        crossdot_u = np.sum(u, axis=-1)        # Shape (Nb, Nb)
        # The cosine similarity. Shape (Nb, Nb)
        cosines = crossdot_u / nrm[:,np.newaxis] / nrm[np.newaxis,:]

        # Cutting edges that do not satisfy the angle condition
        # with any other edge incident to vertex 'i'.
        best_cosine = cosines.min(axis=1)    # Shape (Nb,)
        cosine_thresh = np.cos((180-max_angle)*np.pi/180)    # negative
        valid_local_idx = np.where(best_cosine < cosine_thresh)[0] 
        valid_global_idx = neighb_global_indices[valid_local_idx]
        copy_edges[valid_global_idx, i] = 1
        copy_edges[i, valid_global_idx] = 1
    return copy_edges


# TODO see if useful in a separate function or if must be written inside _compute_labels
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
    return np.stack(distances, axis=-1)

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

def _compute_models_in_cc(
        centroids_cc: np.ndarray
) -> Tuple[np.ndarray, List[SegmentElectrodeModel]]:
    """Computes the optimal group of models from the given set of points.
    A group of model is considered optimal if it minimizes the number of
    models used while ensuring a sufficient fit to the data.

    ### Input:
    - centroids_cc: the array of all 3D coordinates from which to compute the
    models. These points represent all the centroids found in a same connected 
    component. Shape (N, 3).

    # Outputs:
    - labels: the classification labels of the centroids with the
    returned models. Shape (N,).
    - models: the optimal group of models. The length is undefined and
    contained in {1, ..., len(centroids_cc)//2}.
    """

    # Computing a tree of the connected component + removing some noise
    dist_matrix = distance_matrix(centroids_cc)    # Shape (N, N)
    edges = _kruskal(dist_matrix)                  # Shape (N, N), boolean

    # TODO debug remove
    if DEBUG_PLOT:
        global plotter
        plotter = pv.Plotter()
        plot_contacts(centroids_cc)
        plot_tree(centroids_cc, edges, "red")

    #edges = _remove_big_angles(edges, centroids_cc, 45)
    _remove_solo_noise(edges)

    # TODO remove debug
    if DEBUG_PLOT:
        plot_tree(centroids_cc, edges, "blue", 2)
        plotter.show()

    # Extracting the id of the leaves in centroids_cc
    degrees = edges.sum(axis=0)
    #assert 0 not in degrees and len(centroids_cc) > 1, "Tree is not connected"
    leaves = np.where(degrees == 1)[0]

    for n_models in range(1, len(leaves)//2 + 1):
        # The list of all groups of all models (represented by vertices)
        # Generated using ChatGPT.
        ## All possible pairs of leaves
        all_pairs = list(combinations(leaves, 2))
        valid_groups: List[Tuple[Tuple]] = []
        for group in combinations(all_pairs, n_models):
            # Ensuring that all leaves in the group are unique
            leaves_in_group = [leaf for pair in group for leaf in pair]
            if len(set(leaves_in_group)) == 2 * n_models:
                valid_groups.append(group)

        # TODO debug remove
        if DEBUG_PRINT:
            print("---" + " "*20)
            print(f"Leaves: {len(leaves)}")
            print(f"Models: {n_models}")
            print(f"Groups: {len(valid_groups)}")

        # For each group of models, compute the models and their score
        best = {
            'model_group': None,
            'score': None,
            'labels': None
        }
        for group in valid_groups:
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

            # Compute score of group of models
            score = _compute_score(models, centroids_cc, labels)
            if best['score'] == None or score > best['score']:
                best = {
                    'model_group': models,
                    'score': score,
                    'labels': labels
                }

        # If a certain threshold is reached, do not compute bigger groups
        #    -> the number of models to use was found
        if DEBUG_PRINT:
            print("Best:", best['score'])
            print("============")

        if best['score'] > SCORE_THRESHOLD:
            # TODO debug remove
            return best['model_group'], best['labels']
    
    warnings.warn("No group of models has reached the "
                  f"defined threshold ({SCORE_THRESHOLD}).\n"
                  "Either the threshold is too high, or something went wrong.\n"
                  f"Returning the best group anyway (score: {best['score']}).")
    return best['labels'], best['model_group']

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
        models_cc, labels_cc = _compute_models_in_cc(centroids_cc)

        # Converting local labels to global labels and adding new models
        all_labels[tags_dcc == dcc_id] = labels_cc + len(all_models)
        all_models += models_cc

    return all_labels, all_models