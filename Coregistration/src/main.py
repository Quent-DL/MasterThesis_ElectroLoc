from pipeline import *
import sys, os


if __name__ != '__main__':
    print("Error: main.py can only be executed, not imported !")
    exit(1)

assert len(sys.argv) == 5

data_dir  = sys.argv[1]
ct        = sys.argv[2]
anat      = sys.argv[3]
mni       = sys.argv[4]

inDir    = os.path.join(data_dir, 'in')
derivDir = os.path.join(data_dir, 'derivatives')
outDir   = os.path.join(data_dir, 'out')
assert os.path.isdir(inDir)
for dir in (derivDir, outDir):
    os.makedirs(dir, exist_ok=True)


############################
# EXTRACTING BRAINS
############################

anat_brain   = os.path.join(derivDir, "T1wBrain.nii.gz")
anat_brain15 = os.path.join(derivDir, "T1wBrain15mm.nii.gz")
ct_brain15   = os.path.join(derivDir, "CTBrain15mm.nii.gz")

# Replaced by call to 'mri_synth_strip' in the bash submission script
#apply_synthstrip(anat, anat_brain, None, False)
#apply_synthstrip(anat, anat_brain15, None, True)
#apply_synthstrip(ct, ct_brain15, None, True)

############################
# MAPPING CT TO ANAT
############################

map_ct_to_anat = os.path.join(outDir, "Mapping_CT_to_T1w.p")
ct_in_anat     = os.path.join(derivDir, "CT_in_T1w.nii.gz")
ct2anat = coregistration(
    staticPath=anat_brain15,
    movingPath=ct_brain15,
    outputMappingFilepath=map_ct_to_anat,
    outputMappedFilepath=ct_in_anat,
    mode='ct2anat'
)
print("ct2anat done.")

############################
# MAPPING ANAT TO MNI
############################

map_anat_to_mni = os.path.join(outDir, "Mapping_T1w_to_MNI.p")
anat_in_mni     = os.path.join(derivDir, "T1w_in_MNI.nii.gz")
anat2mni = coregistration(
    staticPath=mni,
    movingPath=anat_brain,
    outputMappingFilepath=map_anat_to_mni,
    outputMappedFilepath=anat_in_mni,
    mode='anat2mni'
)
print("anat2mni done.")

############################
# MAPPING CT TO MNI
############################

mappings = [ct2anat, anat2mni]
ct_in_mni = os.path.join(outDir, "CT_in_MNI.nii.gz")
applySuccessiveMappings(
    mappings,
    movingPath=ct,
    staticPath=mni,
    outputMappedFilepath=ct_in_mni
)
print("ct2mni done !")
