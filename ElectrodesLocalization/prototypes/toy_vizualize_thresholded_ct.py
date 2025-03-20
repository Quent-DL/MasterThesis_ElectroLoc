import pyvista as pv
import nibabel as nib


def main():
    input_path   = "D:\\QTFE_local\\Python\\ElectrodesLocalization\\sub04\\in\\CT.nii.gz"
    vol = nib.load(input_path).get_fdata()

    plotter = pv.Plotter()

    # Plotting the volume
    mesh_vol = pv.wrap(vol)
    mesh_vol.cell_data['intensity'] = vol[:-1, :-1, :-1].flatten(order='F')

    # Callback to update the volume
    def update_plot(value):
        thresh_mesh = mesh_vol.threshold(value=value, scalars='intensity')
        plotter.add_mesh(thresh_mesh, cmap='Blues', scalars='intensity', opacity=0.5)

    plotter.add_slider_widget(update_plot, (vol.min(), vol.max()), value=vol.min())
    plotter.show()


if __name__ == '__main__':
    main()