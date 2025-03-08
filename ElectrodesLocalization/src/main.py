import contacts_isolation
import segmentation
import plot
from utils import NibCTWrapper, log
import os


def main():
    # TODO replace: hyperparameters
    ELECTRODE_THRESHOLD = 2500
    n_electrodes = 5
    ct_shape = (512,512,256)

    # Inputs
    input_dir         = "D:\\QTFE_local\\Python\\ElectrodesLocalization\\sub04\\in"
    ct_path           = os.path.join(input_dir, "CT.nii.gz")
    ct_brainmask_path = os.path.join(input_dir, "CTMask.nii.gz")

    # Loading the data
    ct_object = NibCTWrapper(ct_path, ct_brainmask_path)

    # Preprocessing
    ct_object.mask &= (ct_object.ct > ELECTRODE_THRESHOLD)
    
    # Fetching the contacts and converting to physical coordinates
    #contacts = contacts_isolation.get_contacts(ct_object)
    contacts = contacts_isolation.get_contacts(None, synthetic=True)
    contacts = ct_object.apply_affine(contacts, 'forward')

    # Segmenting contacts into electrodes
    electrodes = segmentation.segment_electrodes(contacts, n_electrodes, ct_shape)

    # Converting contacts back to voxel coordinates
    for e in electrodes:
        e.contacts = ct_object.apply_affine(e.contacts, 'inverse')

    # Plotting results
    pv_plotter = plot.plot_colored_electrodes(electrodes)
    pv_plotter.show()



if __name__ == '__main__':
    main()