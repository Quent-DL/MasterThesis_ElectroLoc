# Local modules
import utils
from utils import log, PipelineOutput
import centroids_extraction
import classification_cc
import postprocessing
from electrode_models import ParabolicElectrodeModel, ElectrodeModel

# External modules
import os
import numpy as np
from typing import List, Tuple, Optional

def preprocess(
        nib_wrapper: utils.NibCTWrapper,
        electrode_threshold: float=2500.0
) -> None:
    nib_wrapper.mask &= (nib_wrapper.ct > electrode_threshold)


def extract_centroids_from_nib(
        nib_wrapper: utils.NibCTWrapper,
        precomp_wrapper: Optional[utils.PrecompWrapper] = None,
        # Optional parameters
        struct_name: str = "slice_cross",
        force_computation: bool = False
) -> Tuple[np.ndarray]:
    if (not force_computation
            and precomp_wrapper is not None
            and precomp_wrapper.can_be_loaded()):
        centroids, tags_dcc = precomp_wrapper.load_precomputed_centroids()
    else:
        centroids, tags_dcc = centroids_extraction.extract_centroids(
                ct_grayscale=nib_wrapper.ct, 
                ct_mask=nib_wrapper.mask, 
                struct=centroids_extraction.get_structuring_element(struct_name)
        )

        # Caching the results
        if precomp_wrapper is not None:
            precomp_wrapper.save_precomputed(centroids, tags_dcc)
    return centroids, tags_dcc


def pipeline(
        nib_wrapper: utils.NibCTWrapper,
        electrodes_info: utils.ElectrodesInfo,
        # Optional optimization
        precomp_wrapper: Optional[utils.PrecompWrapper] = None,
        # Hyperparameters
        electrode_threshold: float = 2500,
        recompute_centroids: bool = False,
        skip_postprocessing: bool = False,
        print_logs: bool = True
) -> Tuple[PipelineOutput, List[ElectrodeModel]]:
    ### Preprocessing
    if print_logs: log("Preprocessing data")
    preprocess(nib_wrapper, electrode_threshold)

    ### Fetching approximate contacts
    if print_logs: log("Extracting contacts coordinates")
    centroids_vox, tags_dcc = extract_centroids_from_nib(
        nib_wrapper, precomp_wrapper, 
        force_computation=recompute_centroids)

    ### Converting contacts to physical coordinates
    centroids_world = nib_wrapper.convert_vox_to_world(centroids_vox)
    electrodes_info.entry_points = nib_wrapper.convert_vox_to_world(
        electrodes_info.entry_points)
    ct_center_world = nib_wrapper.convert_vox_to_world(
        np.array(nib_wrapper.ct.shape)/2)
    
    ### Segmenting contacts into electrodes
    if print_logs: log("Classifying contacts to electrodes")
    labels, models = classification_cc.classify_centroids(
        centroids_world, tags_dcc)
    
    ### Postprocessing
    if not skip_postprocessing:
        if print_logs: log("Post-processing results")
        contacts_world, labels, contacts_ids, models = postprocessing.postprocess(
            centroids_world, 
            labels, 
            ct_center_world, 
            models, 
            electrodes_info,
            model_cls=ParabolicElectrodeModel)
    else:
        contacts_world = centroids_world
        contacts_ids = postprocessing._get_electrodes_contacts_ids(
            contacts_world, labels, ct_center_world)
        
    ### Assembling the output
    output = PipelineOutput(
        nib_wrapper.convert_world_to_vox(contacts_world),
        contacts_world,
        labels,
        contacts_ids)
        
    return output, models


def pipeline_from_paths(
    ct_path: str,
    electrodes_info_path: str,
    ct_brainmask_path: Optional[str] = None,
    precomputed_centroids_path: Optional[str] = None,
    print_logs: bool = True,
    **kwargs
) -> Tuple[PipelineOutput, List[ElectrodeModel]]:
    ### Loading the data
    if print_logs: log("Loading data")
    nib_wrapper = utils.NibCTWrapper(ct_path, ct_brainmask_path)
    electrodes_info = utils.ElectrodesInfo(electrodes_info_path)
    precomp_wrapper = utils.PrecompWrapper(precomputed_centroids_path)

    return pipeline(nib_wrapper, electrodes_info, precomp_wrapper, 
                    print_logs=print_logs, **kwargs)