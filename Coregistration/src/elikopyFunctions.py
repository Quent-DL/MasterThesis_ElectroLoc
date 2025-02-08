import os
import numpy as np
import nibabel as nib
from dipy.viz import regtools
from dipy.segment.mask import applymask
from dipy.io.image import load_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric

def getTransform(static_path, moving_path, moving_mask_path=None, onlyAffine=False, shear=True,
                 diffeomorph=True, sanity_check=False, affine_map=None):
    '''


    Parameters
    ----------
    static_volume : 3D array of static volume
    moving_volume : 3D array of moving volume
    shear : if True then no Affine transform
    diffeomorph : if False then registration is only affine
    sanity_check : if True then prints figures

    Returns
    -------
    mapping : transform operation to send moving_volume to static_volume space

    '''

    static, static_affine = load_nifti(static_path)
    static_grid2world = static_affine

    moving, moving_affine = load_nifti(moving_path)
    moving_grid2world = moving_affine

    if moving_mask_path is not None:
        mask, mask_affine = load_nifti(moving_mask_path)
        moving = applymask(moving, mask)

    # Affine registration -----------------------------------------------------

    if sanity_check or onlyAffine:

        if affine_map is None:
            affine_map = np.eye(4)
        affine_map = AffineMap(affine_map,
                               static.shape, static_grid2world,
                               moving.shape, moving_grid2world)
        
        if sanity_check:
        
            resampled = affine_map.transform(moving)

            regtools.overlay_slices(static, resampled, None, 0,
                                    "Static", "Moving", "resampled_0.png")
            regtools.overlay_slices(static, resampled, None, 1,
                                    "Static", "Moving", "resampled_1.png")
            regtools.overlay_slices(static, resampled, None, 2,
                                    "Static", "Moving", "resampled_2.png")

        if onlyAffine:
            return affine_map

    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)

    # !!! TODO: tweak hyperparameters
    metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=c_of_mass.affine)

    transform = RigidTransform3D()
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=translation.affine)

    transform = AffineTransform3D()
    if shear:
        affine = affreg.optimize(static, moving, transform, params0,
                                static_grid2world, moving_grid2world,
                                starting_affine=rigid.affine)
    else:
        affine = rigid

    # Diffeomorphic registration --------------------------

    if diffeomorph:

        metric = CCMetric(3)

        level_iters = [10, 10, 5]
        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

        mapping = sdr.optimize(static, moving, static_affine, moving_affine,
                               affine.affine)

    else:

        mapping = affine

    if sanity_check:
        transformed = mapping.transform(moving)
        # transformed_static = mapping.transform_inverse(static)

        regtools.overlay_slices(static, transformed, None, 0,
                                "Static", "Transformed", "transformed.png")
        regtools.overlay_slices(static, transformed, None, 1,
                                "Static", "Transformed", "transformed.png")
        regtools.overlay_slices(static, transformed, None, 2,
                                "Static", "Transformed", "transformed.png")

    return mapping


def applyTransform(
    input_filepath: str,
    mappings: list, 
    mask_moving_filepath: str=None, 
    binary: bool=False,
    inverse: bool=False, 
    static_filepath: str='', 
    output_path: str='', 
    mask_static_filepath: str=None, 
    #static_fa_file: str=''
):
    '''


    Parameters
    ----------
    input_file : TYPE
        DESCRIPTION.
    mapping : TYPE
        DESCRIPTION.
    static_filepath : TYPE, optional
        Only necessary if output_path is specified. The default is ''.
        Static file, to retrieve the affine information and copy them into the new file
    output_path : TYPE, optional
        If entered, saves result at specified location. The default is ''.
    binary : TYPE, optional
        DESCRIPTION. The default is False.
    inverse : TYPE, optional
        DESCRIPTION. The default is False.
    mask_static : TYPE, optional
        Path to the mask of the static file

    Returns
    -------
    transformed : TYPE
        DESCRIPTION.

    '''
    print("Applying transform to", input_filepath)
    moving = nib.load(input_filepath)
    moving_data = moving.get_fdata()
    print("Moving data shape:", moving_data.shape)

    if mask_moving_filepath is not None:
        mask, mask_affine = load_nifti(mask_moving_filepath)
        moving_data = applymask(moving_data, mask)

    transformed = moving_data
    for mapping in mappings:
        if inverse:
            transformed = mapping.transform_inverse(transformed)
        else:
            transformed = mapping.transform(transformed)

    if binary:
        transformed[transformed > .5] = 1
        transformed[transformed <= .5] = 0

    # TODO: ce code fait quoi ???
    if len(output_path) > 0:
        static = nib.load(static_filepath)

        # static_fa = nib.load(static_fa_file)   TODO remove because static_fa_file no longer parameter

        if mask_static_filepath is not None:
            mask_static_data, mask_static_affine = load_nifti(mask_static_filepath)
            transformed = applymask(transformed, mask_static_data)

        #out = nib.Nifti1Image(transformed, static.affine, header=static_fa.header)
        out = nib.Nifti1Image(transformed, static.affine)
        out.to_filename(output_path)

    else:
        return transformed