# Local modules
import utils
from utils import log
import contacts_isolation
import segmentation_multimodel
import postprocessing
import validation
import plot

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

    # Inputs for algorithm
    data_dir          = ("D:\\QTFE_local\\Python\\ElectrodesLocalization\\"
                         f"data\\{path_suffix}")
    ct_path           = os.path.join(data_dir, "in\\CT.nii.gz")
    ct_brainmask_path = os.path.join(data_dir, "in\\CTMask.nii.gz")
    contacts_path     = os.path.join(data_dir, "derivatives\\raw_contacts.csv")
    elec_info_path    = os.path.join(data_dir, "in\\electrodes_info.csv")
    output_path       = os.path.join(data_dir, "out\\electrodes.csv")
    # Inputs for validation
    ground_truth_path = ("D:\\QTFE_local\\Python\\ElectrodesLocalization\\"
                        f"data_ground_truths\\{path_suffix}\\ground_truth.csv")

    # Loading the data
    log("Loading data")
    ct_object = utils.NibCTWrapper(ct_path, ct_brainmask_path)
    electrodes_info = utils.ElectrodesInfo(elec_info_path)

    # Preparing the object to store and save the output
    output_csv = utils.OutputCSV(output_path, raw_contacts_path=contacts_path)

    # Preprocessing
    log ("Preprocessing data")
    ct_object.mask &= (ct_object.ct > ELECTRODE_THRESHOLD)
    
    # Fetching approximate contacts
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
    
    # Converting contacts to physical coordinates
    contacts = ct_object.convert_vox_to_world(contacts)
    electrodes_info.entry_points = ct_object.convert_vox_to_world(
        electrodes_info.entry_points)
    ct_center_world = ct_object.convert_vox_to_world(
        np.array(ct_object.ct.shape)/2)

    # Segmenting contacts into electrodes
    log("Classifying contacts to electrodes")
    labels, models = segmentation_multimodel.segment_electrodes(
        contacts, N_ELECTRODES)

    # Assigning an id to all contacts of each electrode, based on depth
    log("Post-processing results")
    old_contacts = contacts        # Saved for plotting purposes
    contacts, labels, contacts_ids, models = postprocessing.postprocess(
        contacts, labels, ct_center_world, models, electrodes_info)
    
    # Retrieving stats about distance error
    log("Validating results")
    _, _, _, _, stats_print = validation.validate_contacts_position_from_file(
        contacts, 
        ground_truth_path,
        ct_object.convert_vox_to_world
    )

    # Converting contacts back to voxel coordinates
    contacts = ct_object.convert_world_to_vox(contacts)
    electrodes_info.entry_points = ct_object.convert_world_to_vox(
        electrodes_info.entry_points)

    # Plotting results
    log("Plotting results")
    pv_plotter = None
    # TODO uncomment
    pv_plotter = plot.plot_binary_electrodes(ct_object.mask, pv_plotter)
    #pv_plotter = plot.plot_ct(ct_object.ct, pv_plotter)
    #pv_plotter = plot.plot_contacts(old_contacts, pv_plotter)
    #pv_plotter = plot.plot_contacts(electrodes_info.entry_points, pv_plotter)
    pv_plotter = plot.plot_colored_electrodes(contacts, labels, pv_plotter)
    pv_plotter = plot.plot_linear_electrodes(
        models, 
        ct_object.convert_world_to_vox, 
        pv_plotter)

    pv_plotter.add_axes()
    pv_plotter.show()

    # Saving results to CSV file
    log("Saving results to CSV file")
    output_csv.save_output(contacts, labels, contacts_ids)

    print(stats_print)


if __name__ == '__main__':
    main()