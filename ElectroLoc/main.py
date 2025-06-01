# Local modules
import utils
from utils import log
import pipeline
import validation
from misc.nib_wrapper import NibCTWrapper
import plot

# External modules
import os
from argparse import ArgumentParser, ArgumentTypeError, Namespace


def generate_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="ElectroLoc",
        description="A fast and efficient algorithm to localize sEEG electrodes in CT scans.",
        add_help=True
    )

    # Positional arguments
    parser.add_argument(
        "ct_file",
        type=str,
        metavar="PATH_CT_NIFTI",
        help="Path to the input CT file."
    )
    parser.add_argument(
        "brain_mask_file",
        type=str,
        metavar="PATH_BRAIN_MASK_NIFTI",
        help="Path to the CT's brain mask file."
    )
    parser.add_argument(
        "electrode_info_file",
        type=str,
        metavar="PATH_ELEC_INFO_CSV",
        help="Path to the electrode information CSV file (i.e., entry points and number of contacts)."
    )

    # Optional arguments
    parser.add_argument(
        "--output_coords_file", "-o",
        type=str,
        metavar="PATH_OUT_COORDS_CSV",
        help="Path to save the output results."
    )
    parser.add_argument(
        "--output_mask_file", "-m",
        type=str,
        metavar="PATH_OUT_MASK_NIFTI",
        help="Path to save the mask of the predicted contacts."
    )
    parser.add_argument(
        "--ct_threshold", "-T",
        type=int,
        metavar="INTEGER",
        default = 2500,
        help=("Minimum threshold applied to the CT image to compute the electrode mask. "
              "Recommended value: 2500 <= B <= 2800. Default: 2500")
    )
    parser.add_argument(
        "--branching_factor", "-B",
        type=int,
        metavar="INTEGER",
        default = 2,
        help=("Branching factor of search graph used to compute electrode models. " 
              "High B gives better results but is significantly slower. "
              "Recommended: 1 <= B <= 3. Default: 2")
    )
    parser.add_argument(
        "--dilation_radius", "-R",
        type=int,
        metavar="INTEGER",
        default = 3,
        help=("Radius of the dilation used to identify intersecting electrodes. "
              "Must be bigger than the distance (in voxels) between two contacts. "
              "Recommended: 3 <= B <= 10 for slices with resolution 512x512. Increase B for higher resolution. "
              "Default: 3.")
    )
    parser.add_argument(
        "--ground_truth_file", "-g",
        type=str,
        metavar="PATH_GROUND_TRUTH_CSV",
        help="Path to the ground truth file. If given, the performance of the algorithm is assessed and printed."
    )
    parser.add_argument(
        "--plot_options", "-p",
        type=str,
        metavar="^[ieckv]+$",
        help=("Output plot options. Use any combination of 'i' (CT image)," 
              "'E' (electrode mask), 'M' (electrode models), 'C' (predicted contacts), or 'V' (validate). " 
              "The validation mode shows false positives and false negatives, and requires a ground truth (-g) to be provided.")
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        dest="verbose",
        help="Enable verbose output mode."
    )

    return parser


def validate_args(args: Namespace) -> None:
    """Checks the validity of the arguments. If one check fails, raise an error with a descriptive message."""

    if args.plot_options is not None:
        # Must contain at least one among ['i', 'e', 'm', 'c', 'v']
        plots_allowed = set("iemcv")
        plots_got = set(args.plot_options)
        if len(plots_got) == 0:
            raise ValueError(f"The --plot argument must contain AT LEAST one letter among the following: {list(plots_allowed)}. Use argument --help for help.")
        if not plots_got.issubset(plots_allowed):
            raise ArgumentTypeError(f"The --plot argument must contain ONLY letters among the following: {list(plots_allowed)}. Use argument --help for help.")
        
        # If "v" is included, then ground truth file (-g) must be provided as well
        if "v" in args.plot_options and args.ground_truth_file is None:
            raise ValueError("Plot option 'v' cannot be included if no ground truth (-g) is provided. Use argument --help for help.")
        
        # Check validity of files
        if args.ct_file is not None and not os.path.exists(args.ct_file):
            raise ValueError(f"Argument {args.ct_file} was specified, but the file was not found.")
        
        if args.brain_mask_file is not None and not os.path.exists(args.brain_mask_file):
            raise ValueError(f"Argument {args.brain_mask_file} was specified, but the file was not found.")
        
        if args.electrode_info_file is not None and not os.path.exists(args.electrode_info_file):
            raise ValueError(f"Argument {args.electrode_info_file} was specified, but the file was not found.")
        
        if args.output_coords_file is not None and not os.path.exists(args.output_coords_file):
            raise ValueError(f"Argument {args.output_coords_file} was specified, but the file was not found.")
        
        if args.output_mask_file is not None and not os.path.exists(args.output_mask_file):
            raise ValueError(f"Argument {args.output_mask_file} was specified, but the file was not found.")
        
        if args.ground_truth_file is not None and not os.path.exists(args.ground_truth_file):
            raise ValueError(f"Argument {args.ground_truth_file} was specified, but the file was not found.")
    

    # TODO add further checks about the arguments, e.g. check that electrode info CSV file actually contains the expected column names



def main(args: Namespace):

    ##############################
    # INPUT FILES

    # Inputs for algorithm
    ct_path           = args.ct_file
    ct_brainmask_path = args.brain_mask_file
    electrodes_info_path = args.electrode_info_file

    # Outputs
    output_positions_path = args.output_coords_file
    output_nifti_path = args.output_mask_file

    # Validation
    ground_truth_path = args.ground_truth_file

    ##############################ct_path
    # THE ALGORITHM ITSELF

    output, models = pipeline.pipeline_from_paths(
        ct_path,
        electrodes_info_path,
        ct_brainmask_path,
        # Hyperparams
        electrode_threshold = args.ct_threshold,
        branching_factor_modeling = args.branching_factor,
        dilation_radius_dcc_extraction = args.dilation_radius,
        print_logs = args.verbose,
        # Debug features
        recompute_centroids = True,
        skip_postprocessing = False,
    )
    
    ##############################
    # SAVING RESULTS

    # Copy of the NibWrapper used in the algorithm, for validation and plotting
    nib_wrapper = NibCTWrapper(ct_path, ct_brainmask_path)
    pipeline.preprocess(nib_wrapper, args.ct_threshold)

    # Saving results to CSV file
    if args.verbose: log("Saving results")
    output.to_csv(
        output_positions_path, 
        index=False,
        float_format=lambda f: round(f, 3))

    # Saving mask
    
    if output_nifti_path is not None:
        contacts_vox = output.get_vox_coordinates()
        
        nib_wrapper.save_contacts_mask(
            output_nifti_path, contacts_vox, 
            0.25*utils.estimate_intercontact_distance(contacts_vox))

    ##############################
    # Validation and plot

    ### Validating using ground truth
    if ground_truth_path is not None:
        # Getting predicted contacts and inter-contact distance
        contacts_world = output.get_world_coordinates()
        intercontact_dist_world = utils.estimate_intercontact_distance(
            contacts_world)

        # Getting ground_truths
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
    if args.plot_options is not None:
        if args.verbose: log("Plotting results")
        contacts_vox = output.get_vox_coordinates()

        plotter = plot.ElectrodePlotter(nib_wrapper.convert_world_to_vox)
        plotter.update_focal_point(contacts_vox.mean(axis=0))

        if "i" in args.plot_options:
            plotter.plot_ct(nib_wrapper.ct, alpha_factor=2)
        if "e" in args.plot_options:
            plotter.plot_ct_electrodes(nib_wrapper.mask)
        if "m" in args.plot_options:
            plotter.plot_electrodes_models(models)
        if "c" in args.plot_options:
            plotter.plot_colored_contacts(contacts_vox, output.get_labels()) 
        if "v" in args.plot_options and args.ground_truth_file is not None:
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
    parser = generate_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    main(args)