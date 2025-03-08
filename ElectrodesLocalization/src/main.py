import contacts_isolation
import segmentation
import plot
from utils import NibCTWrapper, log
import os
from numpy import savetxt, loadtxt


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

    # Loading the data
    log("Loading data")
    ct_object = NibCTWrapper(ct_path, ct_brainmask_path)

    # Preprocessing
    log ("Preprocessing data")
    ct_object.mask &= (ct_object.ct > ELECTRODE_THRESHOLD)
        
    # Fetching the contacts and converting to physical coordinates
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
    contacts = ct_object.apply_affine(contacts, 'forward')

    # Segmenting contacts into electrodes
    log("Classifying contacts to electrodes")
    electrodes = segmentation.segment_electrodes(contacts, n_electrodes, ct_shape)

    # Converting contacts back to voxel coordinates
    for e in electrodes:
        e.contacts = ct_object.apply_affine(e.contacts, 'inverse')

    # Plotting results
    log("Plotting results")
    pv_plotter = plot.plot_colored_electrodes(electrodes)

    if not DEBUG_USE_SYNTH:
        plot.plot_binary_electrodes(ct_object.mask, pv_plotter)
        plot.plot_ct(ct_object.ct, pv_plotter)

    pv_plotter.show()


if __name__ == '__main__':
    main()