# Basic imports and loading image
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ct_path = "D:\\QTFE_local\\Python\\ElectrodesLocalization\\sub11\\in\\CT.nii.gz"
ct = nib.load(ct_path).get_fdata()

# Hard threshold based on:
# https://radiopaedia.org/articles/hounsfield-unit

threshold_houndfields = 2800
ct_thresh = ct[ct > threshold_houndfields]
#plt.hist(ct_thresh.ravel(), bins=40)

binary_img = (ct > threshold_houndfields).astype(dtype=np.int8)


# Visualising results in 3D

import pyvista as pv

def Visualize3D(image:np.array):

    grid = pv.ImageData()

    grid.dimensions = np.array(image.shape) + 1
    grid.cell_data['values'] = image.flatten(order='F')
    plotter = pv.Plotter()
    plotter.add_volume(grid, cmap="gray", opacity=[0,0.045])

    from matplotlib.colors import LinearSegmentedColormap ######################
    from matplotlib import cm

    background='white'
    color_map='Reds'
    voxel=True    # False = smooth display

    datapv = pv.wrap(image)
    datapv.cell_data['labels'] = image[:-1, :-1, :-1].flatten(order='F')

    vol = datapv.threshold(value=1, scalars='labels')
    mesh = vol.extract_surface()
    smooth = mesh.smooth_taubin(n_iter=12)       # smoothing the rendering

    if voxel:
        #vol.plot(cmap=color_map, background=background,
        #            scalars='labels')
        plotter.add_mesh(vol)
    else:
        #smooth.plot(cmap=color_map, background=background,
        #            scalars='labels')
        plotter.add_mesh(smooth)

    plotter.show()


Visualize3D(binary_img)