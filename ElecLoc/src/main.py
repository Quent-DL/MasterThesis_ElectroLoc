# Local modules
import utils
from utils import log
import pipeline
import validation
import plot

# External modules
import os


def main():
    DO_PLOT = True
    subId = "sub11"

    # TODO replace hyperparameters
    ELECTRODE_THRESHOLD = 2500

    ##############################
    # INPUT FILES

    # Inputs for algorithm
    data_dir          = ("D:\\QTFE_local\\Python\\ElecLoc\\"
                         f"data\\{subId}")
    ct_path           = os.path.join(data_dir, "in\\CT.nii.gz")
    ct_brainmask_path = os.path.join(data_dir, "in\\CTMask.nii.gz")
    precomputed_centroids_path = os.path.join(data_dir, "derivatives\\raw_contacts.csv")
    electrodes_info_path    = os.path.join(data_dir, "in\\electrodes_info.csv")
    # Outputs
    output_positions_path = os.path.join(data_dir, "out\\electrodes.csv")
    output_nifti_path = os.path.join(data_dir, "out\\electrodes_mask.nii.gz")
    # Validation
    ground_truth_path = os.path.join("D:\\QTFE_local\\Python\\ElecLoc\\"
                                     f"data_ground_truths\\{subId}\\ground_truth.csv")

    ##############################
    # THE ALGORITHM ITSELF

    output, models = pipeline.pipeline_from_paths(
        ct_path,
        electrodes_info_path,
        ct_brainmask_path,
        precomputed_centroids_path,
        # Hyperparams
        electrode_threshold = ELECTRODE_THRESHOLD,
        # Features
        recompute_centroids = False,
        skip_postprocessing = False,
        print_logs = True
    )
    
    ##############################
    # SAVING RESULTS
    nib_wrapper = utils.NibCTWrapper(ct_path, ct_brainmask_path)
    pipeline.preprocess(nib_wrapper, ELECTRODE_THRESHOLD)

    # Saving results to CSV file
    log("Saving results")
    output.to_csv(
        output_positions_path, 
        index=False,
        float_format=lambda f: round(f, 3))

    # Saving mask
    contacts_vox = output.get_vox_coordinates()
    intercontact_dist_vox = utils.estimate_intercontact_distance(
        contacts_vox)
    
    nib_wrapper.save_contacts_mask(
        output_nifti_path, contacts_vox, 0.25*intercontact_dist_vox)

    ##############################
    # VISUAL DEBUG ZONE
    contacts_world = output.get_world_coordinates()
    intercontact_dist_world = utils.estimate_intercontact_distance(
        contacts_world)

    is_ground_truth_available = (ground_truth_path is not None
                              and os.path.exists(ground_truth_path))

    ### Printing results
    if is_ground_truth_available:
        ground_truth = utils.PipelineOutput.from_csv(
            ground_truth_path)
        ground_truth_vox = ground_truth.get_vox_coordinates()
        ground_truth_world = nib_wrapper.convert_vox_to_world(
            ground_truth_vox)
    
        (matched_dt_idx, matched_gt_idx, excess_dt_idx, holes_gt_idx, 
                rlvt_distances) = validation.check_contacts_positions(
                    contacts_world,
                    ground_truth_world,
                    0.75 * intercontact_dist_world)
        validation.print_position_results(
            len(contacts_world), len(ground_truth_world), len(matched_dt_idx),
            len(excess_dt_idx), len(holes_gt_idx), rlvt_distances)

    ### Plotting results in voxel space
    if DO_PLOT:
        log("Plotting results")
        plotter = plot.ElectrodePlotter(nib_wrapper.convert_world_to_vox)
        plotter.update_focal_point(contacts_vox.mean(axis=0))
        
        #plotter.plot_ct(nib_wrapper.ct)
        plotter.plot_ct_electrodes(nib_wrapper.mask)
        plotter.plot_electrodes_models(models)
        plotter.plot_colored_contacts(contacts_vox, output.get_labels()) 
        if is_ground_truth_available:
            plotter.plot_contacts(
                contacts_vox[excess_dt_idx], 
                color=(0,0,0), size_multiplier=2.5)
            plotter.plot_contacts(
                ground_truth_vox[holes_gt_idx], 
                color=(255,255,255), size_multiplier=2.5)
            plotter.plot_differences(contacts_vox[matched_dt_idx], 
                                    ground_truth_vox[matched_gt_idx])

        plotter.show()


if __name__ == '__main__':
    main()