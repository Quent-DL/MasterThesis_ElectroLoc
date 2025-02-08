from elikopyFunctions import getTransform, applyTransform

import os
import pickle

from typing import Literal, List, Optional

from dipy.align.imaffine import AffineMap
from dipy.align.imwarp import DiffeomorphicMap

###################################
#
###################################


def apply_synthstrip(
        inputPath             : str | os.PathLike,
        outputStrippedFilepath: str | os.PathLike,
        outputMaskFilepath    : Optional[str | os.PathLike] = None,
        looseMargin: bool = False
) -> None:
      margin = "-b 15" if looseMargin else ""
      mask  = f"-m {outputMaskFilepath}" if outputMaskFilepath is not None else ""
      os.system(f"srun mri_synth_strip \
                -i {inputPath} \
                -o {outputStrippedFilepath} \
                {margin} \
                {mask}")


def coregistration(
        staticPath           : str | os.PathLike, 
        movingPath           : str | os.PathLike, 
        outputMappingFilepath: str | os.PathLike,
        outputMappedFilepath : str | os.PathLike,
        mode: Literal['ct2anat', 'anat2mni']
) -> AffineMap | DiffeomorphicMap:
    # Performing the coregistration to obtain the mapping
    assert mode in ['ct2anat', 'anat2mni']
    if mode == 'ct2anat':
        kwargs = {'shear': False, 'diffeomorph': False}
    else:
        kwargs = {'shear': True, 'diffeomorph': True}

    mapping = getTransform(staticPath, movingPath, **kwargs)

    # Saving the mapping
    with open(outputMappingFilepath, 'wb') as handle:
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Applying the mapping onto the moving volume + saving result
    mapped = applyTransform(
        input_filepath=movingPath,
        mappings=[mapping],
        mask_moving_filepath=None,
        binary=False,
        inverse=False,
        static_filepath=staticPath,
        output_path=outputMappedFilepath,
        mask_static_filepath=None,
    )

    return mapping
    

def applySuccessiveMappings(
        mappings            : List[AffineMap | DiffeomorphicMap],
        movingPath          : str | os.PathLike,
        staticPath          : str | os.PathLike,
        outputMappedFilepath: str | os.PathLike
):
    mapped = applyTransform(
        input_filepath=movingPath,
        mappings=mappings,
        mask_moving_filepath=None,
        binary=False,
        inverse=False,
        static_filepath=staticPath,
        output_path=outputMappedFilepath,
        mask_static_filepath=None,
    )

