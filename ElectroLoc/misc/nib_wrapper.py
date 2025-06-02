import nibabel as nib
import numpy as np
from typing import Optional, Literal, Tuple
from pathlib import Path

class NibCTWrapper:
    def __init__(self, ct_path: str, ct_brainmask_path: str=None):
        # Thresholding and skull-stripping CT
        nib_ct = nib.load(ct_path)

        # The arrays of the CT and brain mask
        self.ct   = nib_ct.get_fdata()
        if ct_brainmask_path is not None:
            self.mask = nib.load(ct_brainmask_path).get_fdata().astype(bool)
        else:
            self.mask = np.ones_like(self.ct, dtype=bool)

        # Considering that voxels may not be square but rectangular
        # (this info is stored in the file's affine matrix)
        # we compute the sigma to apply to each axis' sigma for a Gaussian
        # to account for those different voxel side lengths
        self.affine = nib_ct.affine

    def get_voxel_size(self):
        return np.abs(np.diag(self.affine[:3,:3]))
    
    def get_center_world(self):
        """Returns the world coordinates of the image center"""
        return self.convert_vox_to_world(np.array(self.ct.shape)/2)

    def get_zoom(self):
        """Returns the zoom to apply to the image for isotropic distances."""
        vox_size = self.get_voxel_size()
        zoom_factor = vox_size / vox_size.min()
        return zoom_factor

    def __apply_affine(
            self,
            coords: np.ndarray, 
            mode: Optional[Literal['forward', 'inverse']] = "forward",
            apply_translation: Optional[bool] = True
    ) -> np.ndarray:
        """TODO write documentation
        
        coords: shape (3,) or (N,3)"""
        assert mode in ['forward', 'inverse']

        # The homogenous transform matrix, shape (4, 4)
        A = self.affine if mode=="forward" else np.linalg.inv(self.affine)
        if not apply_translation:
            A[:3,3] = 0    # removing translation coefficients

        if len(coords.shape) == 2:
            # Corods of shape (N, 3)
            # Adding 1's to get homogeneous coordinates
            ones = np.ones((coords.shape[0], 1), dtype=np.float32)
            # Shape (4, N)  
            hmg_coords = np.concatenate([coords, ones], axis=1).T
            # Getting rid of the homogeneous 1's + reshaping to (N, 3)
            return (A @ hmg_coords)[:3,:].T
        else:
            # Coords of shape (3,)
            ones = np.ones((coords.shape[0], 1), dtype=np.float32)
            hmg_coords = np.append(coords, 1).reshape((4,1))   # Shape(4, 1)
            # Getting rid of homogeneous 1 + reshaping to (3,)
            return (A @ hmg_coords)[:3].reshape((3,))
        
    def convert_vox_to_world(
            self, 
            vox_coords: np.ndarray,
            apply_translation: Optional[bool] = True
    ) -> np.ndarray:
        """TODO write documentation. Input shape (3,) or (N, 3)"""
        return self.__apply_affine(vox_coords, 'forward', apply_translation)
    
    def convert_world_to_vox(
            self, 
            vox_coords: np.ndarray,
            apply_translation: Optional[bool] = True
    ) -> np.ndarray:
        """TODO write documentation. Input shape (3,) or (N, 3)"""
        return self.__apply_affine(vox_coords, 'inverse', apply_translation)

    def save_contacts_mask(self, path: str, 
                           contacts: np.ndarray, r:float) -> np.ndarray:
        """Creates a binary mask in voxel space of all the given contacts,
        represented by small spheres, and saves it to the specified path
        as a '.nii.gz' file.
        
        The spheres are isotropic in physical space, and may thus appear
        as anisotropic in voxel space.
        
        ### Inputs:
        - path: the path where to save the resulting '.nii.gz' file.
        - contacts: the coordinates (in voxel space) where to add masked 
        spheres. Shape (N, 3).
        - r: the radius of the spheres (voxel space).
        
        ### Output:
        - mask: the generated binary mask which highlights the positions
        of the contacts. Boolean array with same shape as 'self.ct'."""

        # TODO fix buggy indices in this function

        # Shape (L, W, H)
        contacts_mask = np.zeros_like(self.ct, dtype=bool)

        def cube_around(shape: Tuple[int], 
                        center: np.ndarray, half_length: int) -> Tuple[int]:
            x_bfr = int(np.ceil(center[0] - max(0, center[0]-half_length)))
            x_aft = int(np.ceil(min(shape[0]-1, center[0]+half_length+1) - center[0]))
            y_bfr = int(np.ceil(center[1] - max(0, center[1]-half_length)))
            y_aft = int(np.ceil(min(shape[1]-1, center[1]+half_length+1) - center[1]))
            z_bfr = int(np.ceil(center[2] - max(0, center[2]-half_length)))
            z_aft = int(np.ceil(min(shape[2]-1, center[2]+half_length+1) - center[2]))
            return x_bfr, x_aft, y_bfr, y_aft, z_bfr, z_aft

        def mask_box(box_c: np.ndarray) -> None:
            """Returns the boolean sphere around voxel c into the flattened mask."""
            # Generated using ChatGPT
            boxL, boxW, boxH = (x_bfr+1+x_aft, y_bfr+1+y_aft, z_bfr+1+z_aft)
            x, y, z = np.meshgrid(
                np.arange(boxL), np.arange(boxW), np.arange(boxH), indexing='ij')
            box_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
            distances = np.linalg.norm(
                self.get_voxel_size()*box_points - box_c, axis=-1)
            mask_c = distances < r
            return mask_c.reshape(boxL, boxW, boxH)

        # Generating binary mask
        for c in contacts:  
            # Optimisation: only considering a box around c to compute the
            # distances => MUCH faster
            # Computing margin between the edges of the box and the center
            c = c.astype(int)
            x_bfr, x_aft, y_bfr, y_aft, z_bfr, z_aft = (
                cube_around(contacts_mask.shape, c, np.ceil(r)))
            center_box = np.array([x_bfr, y_bfr, z_bfr])
            # Computing mask within the box, then adding 
            contacts_mask[c[0]-x_bfr: c[0]+x_aft+1, 
                          c[1]-y_bfr: c[1]+y_aft+1, 
                          c[2]-z_bfr: c[2]+z_aft+1] |= mask_box(center_box)
            
        # Simply return if no path given
        if path is None: return contacts_mask

        # Saving mask into Nifti file, creating parent directories if necessary
        img = nib.nifti1.Nifti1Image(contacts_mask.astype(int), 
                                     self.affine, dtype=np.uint8)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, path)
