import numpy as np
from scipy.ndimage import (binary_erosion, binary_propagation,
                           label, center_of_mass)
from utils import log
from typing import Tuple
import typing


def __get_structuring_element(type='cross'):
    if type == 'cube':
        return np.ones((3,3,3))
    elif type == 'cross':
        return np.array([
            [
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ],
            [
                [0,1,0],
                [1,1,1],
                [0,1,0]
            ],
            [
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ]
        ])
    else:
        raise ValueError(
            f"Structuring element type must be 'cube' or 'cross'. Got: {type}")
    

def __binary_ultimate_erosion(image: np.ndarray, struct: np.ndarray):
    """Performs ultimate erosion on an N-dimensional binary image.
    Ultimate erosion is a technique that performs successive erosions on all
    objects of an image until they are reduced to atomic connected components, 
    which would disappear if one more erosion was performed on them. The output
    of the ultimare erosion algorithm is the union of all those connected
    components.
    
    ### Inputs:
    - image: an array of dtype 'bool' of arbitrary dimensions and shape.
    - struct: a binary array with number of dimensions identical to 'image' 
    that represents the structuring element used in the erosion and 
    reconstruction parts of the binary erosion algorithm.
    
    ### Output:
    - result: a binary array with the same shape and dtype identical as 'image'
    which contains the result of the ultimate erosion applied on 'image' with
    'struct' as structuring element.
    """
    # Ref: Proakis, G. John
    result = np.zeros_like(image)

    while image.sum() > 0:
        eroded        = binary_erosion(image, struct)
        reconstructed = binary_propagation(eroded, struct, image)

        # The xor is equivalent to substracting 'reconstructed' from 'image' 
        # if they only consist of 0 and 1's, knowing that reconstructed < image
        result |= np.logical_xor(image, reconstructed)
        image = eroded

    return result


def __get_box(mask: np.ndarray, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Extracts the smallest box that fully contains all the 1's in the given
    binary mask and returns it. That box is also be applied to any additional
    array given.

    We define as "box" a set of indices [lx:ux, ly:uy, lz:uz] used to
    extract sub-arrays from bigger ones. We compute it by finding the biggest
    lx, ly, lz and the smallest ux, uy, uz such that mask[lx:ux, ly:uy, lz:uz]
    contains all the 1's present in 'mask'.

    ### Inputs:
    - mask: the input mask from which the box is computed. Shape (W, L, H).
    - arrays: other arrays to which the box must be applied. Must all have
    the same shape as 'mask'.

    ### Outputs:
    The output is the tuple (offset, *boxes) where:
    - offset: the coordinates of the 'bottom' corner of the extracted box.
    An item at indice [i, j, k] in the resulting box can be found at indices
    [i+offset[0], j+offset[1], k+offset[2]] in the original input array.
    (see below). Array with shape (3,).
    - *boxes: the result when extracting a box from 'mask' and applying it to
    all the given arrays. The length of 'boxes' matches the number of input
    arrays ('mask' included). Each item in 'boxes' is an array with identical
    shape (W', L', H') such that W'<=W, L'<=L, and H'<=H. 

"""
    # Computing the lower and upper bound of the coordinates of the useful box
    lx = np.where(mask.any(axis=(1, 2)))[0].min()
    ux = np.where(mask.any(axis=(1, 2)))[0].max() + 1
    ly = np.where(mask.any(axis=(0, 2)))[0].min()
    uy = np.where(mask.any(axis=(0, 2)))[0].max() + 1
    lz = np.where(mask.any(axis=(0, 1)))[0].min()
    uz = np.where(mask.any(axis=(0, 1)))[0].max() + 1

    offset = np.array([lx, ly, lz], dtype=np.float32)

    # Truncating the arrays to boxes
    boxes = []
    for arr in [mask, *arrays]:
        box_arr = arr[lx:ux, ly:uy, lz:uz]
        boxes.append(box_arr)
    return offset, *boxes


def compute_contacts_centers(
        ct_grayscale: np.ndarray,
        ct_mask: np.ndarray,
        struct: np.ndarray
) -> np.ndarray:
    """Extracts the coordinates of the electrodes contacts (centroids) from the 
    CT image.First, atomic connected components are extracted from the mask 
    using theultimate erosion algorithm. Then, the center of mass of each 
    connected component (weighted by the values in 'ct_grayscale') are computed.
    
    ### Inputs:
    - ct_grayscale: an array of shape (L, M, N) that contains the full 
    grayscale CT scan.
    - ct_mask: a binary array of shape identical to 'ct_grayscale' that 
    contains a mask of the electrode contacts in the CT scan.
    - struct: a binary array of dtype 'bool' and of shape (I, J, K) that 
    contains the structuring element used by the ultimate erosion algorithm.
    
    ### Output:
    - contacts_com: an array of shape (NC,3) that contains the 3D coordinates 
    of all NC contacts identified."""

    # Identifies the connected components of the mask.
    # This is done so that, for each connected component, we only
    # manipulate the smallest 3D arry that contains it, as to reduce
    # the complexity of the algorithm
    # (instead of dealing with the whole array)
    #
    # Normally, one connected component should represent one contact, a section
    # of electrode, or sections of touching electrodes.
    connected_comps_labels, n_comps = label(ct_mask)

    centroids = []
    for cc_id in range(1, n_comps+1):    # For each connected component (CC)
        # Keeping only the smallest array that fully contains the CC (= box)
        comp = (connected_comps_labels == cc_id)
        cc_offset, box_cc, box_gray = __get_box(comp, ct_grayscale)
        # Computing ultimate erosion in that reduced array (box)
        ult_regions = __binary_ultimate_erosion(box_cc, struct)
        # Computing the center of mass (= COM) of all ultimately reduced 
        # regions in the CC
        ult_regions_labels, nb_regions = label(ult_regions)
        centers = center_of_mass(
            box_gray, 
            labels=ult_regions_labels, 
            index=range(1, nb_regions+1))
        # Adding the COM's to the results, while accounting for the box's
        # offset within the big initial array
        centroids.append(np.array(centers) + cc_offset)

    all_centroids = np.concatenate(centroids, dtype=np.float32)

    # Sorting along arbitrary criterion to guarantee non-stochasting 
    # ordering of the contacts
    return all_centroids[np.lexsort(keys=all_centroids.T)]