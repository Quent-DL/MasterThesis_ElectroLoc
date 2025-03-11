import contacts_isolation
import segmentation
import plot
from utils import NibCTWrapper, log
import utils
import os
from numpy import savetxt, loadtxt
import pandas as pd
import numpy as np


def main():
    # TODO replace: hyperparameters
    DEBUG_USE_SYNTH = True
    ELECTRODE_THRESHOLD = 2500
    n_electrodes = 8
    ct_shape = (512,512,256)

    # Inputs
    input_dir         = "D:\\QTFE_local\\Python\\ElectrodesLocalization\\sub11\\in"
    ct_path           = os.path.join(input_dir, "CT.nii.gz")
    ct_brainmask_path = os.path.join(input_dir, "CTMask.nii.gz")
    contacts_path     = os.path.join(input_dir, "contacts.txt")
    output_dir        = "D:\\QTFE_local\\Python\\ElectrodesLocalization\\sub11\\out"
    output_csv_path   = os.path.join(output_dir, "electrodes.csv")

    # Loading the data
    log("Loading data")
    ct_object = NibCTWrapper(ct_path, ct_brainmask_path)

    # Preprocessing
    log ("Preprocessing data")
    ct_object.mask &= (ct_object.ct > ELECTRODE_THRESHOLD)
        
    # Fetching approximate contacts
    log("Extracting contacts coordinates")
    if not DEBUG_USE_SYNTH:
        if not os.path.exists(contacts_path):
            contacts = contacts_isolation.get_contacts(ct_object)
            # TODO CODING REMOVE: caching the contacts
            savetxt(contacts_path, contacts)
        else:
            contacts = loadtxt(contacts_path)
    else:
        contacts = contacts_isolation.get_contacts(None, synthetic=True)

    # Converting contacts to physical coordinates
    contacts = ct_object.apply_affine(contacts, 'forward')

    # Segmenting contacts into electrodes
    log("Classifying contacts to electrodes")
    labels = segmentation.segment_electrodes(contacts, n_electrodes, ct_shape)

    # Assigning an id to all contacts of each electrode, based on depth
    ct_center_physical = ct_object.apply_affine(np.array(ct_shape)/2, 'forward')
    contacts_ids = utils.get_electrodes_contacts_ids(
            contacts, labels, ct_center_physical)

    # Converting contacts back to voxel coordinates
    contacts = ct_object.apply_affine(contacts, 'inverse')

    # Plotting results
    log("Plotting results")
    pv_plotter = None
    if not DEBUG_USE_SYNTH:
        pv_plotter = plot.plot_binary_electrodes(ct_object.mask, pv_plotter)
        plot.plot_ct(ct_object.ct, pv_plotter)
    pv_plotter = plot.plot_colored_electrodes(contacts, labels, pv_plotter)
    pv_plotter.show()

    # Saving results to CSV file
    df_content = {
        'CT voxel x': contacts[:,0],
        'CT voxel y': contacts[:,1],
        'CT voxel z': contacts[:,2],
        'Electrode id': labels,
        'Contact id': contacts_ids
    }
    df = pd.DataFrame(df_content)
    df.sort_values(by=['Electrode id', 'Contact id'], 
                   axis='index', inplace=True)
    df.to_csv(output_csv_path, index=False)

if __name__ == '__main__':
    main()