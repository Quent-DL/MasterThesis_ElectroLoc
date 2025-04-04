# Local modules
import utils
from utils import log
import contacts_isolation
import segmentation_multimodel
import postprocessing
import validation
import plot

from electrode_models import LinearElectrodeModel, ParabolicElectrodeModel

# External modules
import os
import numpy as np

def main():
    # TODO Remove: synthetic data
    DEBUG_USE_SYNTH = False
    path_suffix = ("sub11" if not DEBUG_USE_SYNTH else "synthetic01")

    # TODO replace hyperparameters
    ELECTRODE_THRESHOLD = 2500
    N_ELECTRODES = (8 if not DEBUG_USE_SYNTH else 5)

    ### Inputs
    # Inputs for algorithm
    data_dir          = ("D:\\QTFE_local\\Python\\ElectrodesLocalization\\"
                         f"data\\{path_suffix}")
    ct_path           = os.path.join(data_dir, "in\\CT.nii.gz")
    ct_brainmask_path = os.path.join(data_dir, "in\\CTMask.nii.gz")
    contacts_path     = os.path.join(data_dir, "derivatives\\raw_contacts.csv")
    elec_info_path    = os.path.join(data_dir, "in\\electrodes_info.csv")
    # Inputs for validation
    ground_truth_path = ("D:\\QTFE_local\\Python\\ElectrodesLocalization\\"
                        f"data_ground_truths\\{path_suffix}\\ground_truth.csv")
    # Output
    output_positions_path = os.path.join(data_dir, "out\\electrodes.csv")
    output_nifti_path = os.path.join(data_dir, "out\\electrodes_mask.nii.gz")

    ### Loading the data
    log("Loading data")
    ct_object = utils.NibCTWrapper(ct_path, ct_brainmask_path)
    electrodes_info = utils.ElectrodesInfo(elec_info_path)

    ### Preparing the object to store and save the output
    output_csv = utils.OutputCSV(output_positions_path, raw_contacts_path=contacts_path)

    ### Preprocessing
    log ("Preprocessing data")
    ct_object.mask &= (ct_object.ct > ELECTRODE_THRESHOLD)
    
    ### Fetching approximate contacts
    log("Extracting contacts coordinates")
    if output_csv.are_raw_contacts_available():
        contacts = output_csv.load_raw_contacts()
    else:
        contacts = contacts_isolation.compute_contacts_centers(
                ct_grayscale=ct_object.ct, 
                ct_mask=ct_object.mask, 
                struct=contacts_isolation.__get_structuring_element('cross')
        )
        # Caching the results
        output_csv.save_raw_contacts(contacts)
    
    ### Converting contacts to physical coordinates
    contacts = ct_object.convert_vox_to_world(contacts)
    electrodes_info.entry_points = ct_object.convert_vox_to_world(
        electrodes_info.entry_points)
    ct_center_world = ct_object.convert_vox_to_world(
        np.array(ct_object.ct.shape)/2)

    ### Segmenting contacts into electrodes
    log("Classifying contacts to electrodes")
    labels, models = segmentation_multimodel.segment_electrodes(
        contacts, N_ELECTRODES, 
        model_cls=LinearElectrodeModel)


    ### Assigning an id to all contacts of each electrode, based on depth
    log("Post-processing results")
    contacts, labels, contacts_ids, models = postprocessing.postprocess(
        contacts, labels, ct_center_world, models, electrodes_info,
        model_cls=ParabolicElectrodeModel)
    intercontact_dist_world = postprocessing.__estimate_intercontact_distance(contacts)
    
    ### Validation: retrieving stats about distance error
    log("Validating results")
    ground_truth = validation.get_ground_truth(ground_truth_path,
                                               ct_object.convert_vox_to_world)
    results = validation.validate_contacts_position(
        contacts,
        ground_truth,
        0.75*intercontact_dist_world,
    )
    (matched_dt_idx, matched_gt_idx, 
        excess_dt_idx, holes_gt_idx, stats_print) = results

    ### Plotting results in voxel space
    log("Plotting results")
    plotter = plot.ElectrodePlotter(ct_object.convert_world_to_vox)
    plotter.update_focal_point(contacts.mean(axis=0))
    #plotter.plot_ct(ct_object.ct)
    plotter.plot_ct_electrodes(ct_object.mask)
    plotter.plot_electrodes(models)
    plotter.plot_colored_contacts(contacts, labels)
    plotter.plot_contacts(contacts[excess_dt_idx], color=(0,0,0), 
                          size_multiplier=2.5)
    plotter.plot_contacts(ground_truth[holes_gt_idx], color=(255,255,255), 
                          size_multiplier=2.5)
    plotter.plot_matches(contacts[matched_dt_idx], 
                         ground_truth[matched_gt_idx])
    plotter.show()

    # Saving results to CSV file
    log("Saving results")
    contacts_vox = ct_object.convert_world_to_vox(contacts)
    intercontact_dist_vox = postprocessing.__estimate_intercontact_distance(contacts)
    output_csv.save_output(contacts_vox, labels, contacts_ids)
    ct_object.save_contacts_mask(
        output_nifti_path, contacts_vox, 0.25*intercontact_dist_vox)

    print(stats_print)


if __name__ == '__main__':
    main()