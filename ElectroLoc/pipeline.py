# Local modules
import utils
from utils import log, PipelineOutput
from misc.nib_wrapper import NibCTWrapper
from misc.electrode_information import ElectrodesInfo
import centroids_extraction
import linear_modeling
import postprocessing
from misc.electrode_models import ParabolicElectrodeModel, ElectrodeModel

# External modules
from typing import List, Tuple, Optional


def preprocess(
        nib_wrapper: NibCTWrapper,
        electrode_threshold: float=2600.0
) -> None:
    nib_wrapper.mask &= (nib_wrapper.ct > electrode_threshold)


def pipeline(
        nib_wrapper: NibCTWrapper,
        electrodes_info: ElectrodesInfo,
        # Hyperparameters
        electrode_threshold: float = 2500.0,     # TODO hyperparameter
        branching_factor_modeling: int = 2,
        dilation_radius_dcc_extraction: int = 3,
        # Debug parameters
        recompute_centroids: bool = False,
        skip_postprocessing: bool = False,
        print_logs: bool = True
) -> Tuple[PipelineOutput, List[ElectrodeModel]]:
    ### Preprocessing
    if print_logs: log("Preprocessing data")
    preprocess(nib_wrapper, electrode_threshold)

    ### Fetching approximate contacts
    if print_logs: log("Extracting contacts coordinates")
    centroids_vox, tags_dcc = centroids_extraction.extract_centroids(
        ct_grayscale=nib_wrapper.ct, 
        electrode_mask=nib_wrapper.mask, 
        struct_name = "slice_cross",
        dcc_dilation_radius=dilation_radius_dcc_extraction)

    ### Converting contacts to physical coordinates
    centroids_world = nib_wrapper.convert_vox_to_world(centroids_vox)
    electrodes_info.entry_points = nib_wrapper.convert_vox_to_world(
        electrodes_info.entry_points)
    
    ### Segmenting contacts into electrodes
    if print_logs: log("Classifying contacts to electrodes")
    labels, models = linear_modeling.classify_centroids(
        centroids_world, 
        tags_dcc, 
        electrodes_info.nb_electrodes,
        branching_factor_modeling)
    
    ### Postprocessing
    # TODO add hyperparameters to function call
    if not skip_postprocessing:
        if print_logs: log("Post-processing results")
        contacts_world, labels, contacts_ids, models = postprocessing.postprocess(
            centroids_world, 
            labels, 
            models, 
            electrodes_info,
            model_recomputation_class=ParabolicElectrodeModel, 
            ct_center=nib_wrapper.get_center_world())
    else:
        contacts_world = centroids_world
        contacts_ids = postprocessing._get_electrodes_contacts_ids(
            contacts_world, labels, nib_wrapper.get_center_world())
        
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
    print_logs: bool = True,
    **kwargs
) -> Tuple[PipelineOutput, List[ElectrodeModel]]:
    ### Loading the data
    if print_logs: log("Loading data")
    nib_wrapper = NibCTWrapper(ct_path, ct_brainmask_path)
    electrodes_info = ElectrodesInfo(electrodes_info_path)

    return pipeline(nib_wrapper, electrodes_info, 
                    print_logs=print_logs, **kwargs)