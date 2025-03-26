import numpy as np
from scipy.ndimage import (binary_erosion, binary_propagation,
                           label, center_of_mass)
from utils import log


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


def __opti_center_of_mass(input, labels, index):
    """This function is an optimized wrapper of the function 
    'scipy.ndimage.center_of_mass' that restricts the size of the input array
    and only keep the smallest relevant box (i.e. the smallest box that 
    contains all occurences of 'index') before feeding it to 
    scipy.ndimage.center_of_mass.
    
    ### Inputs:
    The inputs are the same as those of scipy.ndimage.center_of_mass. Arrays
    'input' and 'labels' must both be of shape (L, M, N).
    
    ### Output:
    coords: an array of shape (3,) that contains the same coordinates as 
    returned by scipy.ndimage.center_of_mass, but computed faster."""
    # Computing the lower and upper bound of the coordinates of the useful box
    mask = (labels == index)
    lx = np.where(mask.any(axis=(1, 2)))[0].min()
    ux = np.where(mask.any(axis=(1, 2)))[0].max() + 1
    ly = np.where(mask.any(axis=(0, 2)))[0].min()
    uy = np.where(mask.any(axis=(0, 2)))[0].max() + 1
    lz = np.where(mask.any(axis=(0, 1)))[0].min()
    uz = np.where(mask.any(axis=(0, 1)))[0].max() + 1

    # Truncating the useful boxes
    input_trunc = input[lx:ux, ly:uy, lz:uz]
    labels_trunc = labels[lx:ux, ly:uy, lz:uz]

    # Computing the center of mass in the truncated box, then adding the offset
    offset = np.array([lx, ly, lz], dtype=np.float32)
    trunc_center = np.array(
        center_of_mass(input_trunc, labels_trunc, index), 
        dtype=np.float32
    )
    return offset + trunc_center


def compute_contacts_centers(
        ct_grayscale: np.ndarray,
        ct_mask: np.ndarray,
        struct: np.ndarray
) -> np.ndarray:
    """Extracts the coordinates of the electrodes contacts from the CT image.
    First, atomic connected components are extracted from the mask using the
    ultimate erosion algorithm. Then, the center of mass of each connected
    component (weighted by the values in 'ct_grayscale') are computed.
    
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
    log("Computing ultimate erosion", erase=True)
    ult_er = __binary_ultimate_erosion(ct_mask, struct)
    labels, n_contacts = label(ult_er)
    
    contacts_com = []
    for i in range(1, n_contacts+1):
        log(f"Contact {i}/{n_contacts}", erase=True)
        contacts_com.append(__opti_center_of_mass(ct_grayscale, labels, i))

    all_contacts = np.stack(contacts_com, dtype=np.float32)

    # Sorting along arbitrary criterion to guarantee non-stochasting 
    # ordering of the contacts
    a = all_contacts[np.lexsort(keys=all_contacts.T)]
    print(a.shape)
    return a