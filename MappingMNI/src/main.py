import pickle
import utils
from dipy.align.imaffine import AffineMap
from dipy.align.imwarp import DiffeomorphicMap
import nibabel as nib

# TODO remove in final version
import hardcoded_data


# Inputs
ct2anat_path  = "D:\\QTFE_local\\Python\\Coregistration\\sub11\\out\\Mapping_CT_to_T1w.p"
anat2mni_path = "D:\\QTFE_local\\Python\\Coregistration\\sub11\\out\\Mapping_T1w_to_MNI.p"
mni_path      = "D:\\QTFE_local\\Python\\data\\MNI.nii.gz"

# TODO remove in final version
ct_path ="D:\\QTFE_local\\Python\\Coregistration\\sub11\\in\\CT.nii.gz"


if __name__ == '__main__':
    # Loading the coordinates
    # TODO replace hardcoded by actual version
    contacts_ct = hardcoded_data.CONTACTS

    # Loading coregistration transforms
    with open(ct2anat_path, 'rb') as handle:
                ct2anat: AffineMap = pickle.load(handle)
    with open(anat2mni_path, 'rb') as handle:
                anat2mni: DiffeomorphicMap = pickle.load(handle)

    contacts_mni = utils.transform_from_ct_to_mni(
            contacts_ct,
            hardcoded_data.ct_vox2world,
            hardcoded_data.anat_vox2world,
            hardcoded_data.mni_vox2world,
            ct2anat,
            anat2mni
    )
    
    # Plotting CT
    ct = nib.load(ct_path).get_fdata()
    plotter = utils.plot_volume(ct)
    utils.plot_contacts(contacts_ct, plotter)
    plotter.show()

    # Plotting MNI
    mni = nib.load(mni_path).get_fdata()
    plotter = utils.plot_volume(mni)
    utils.plot_contacts(contacts_mni, plotter)
    plotter.show()   
