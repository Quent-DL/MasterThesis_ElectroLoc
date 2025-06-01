import pandas as pd

class ElectrodesInfo:
    _VOX_KEYS = ['vox_x', 'vox_y', 'vox_z']

    def __init__(self, path):
        """Initialize an instance from the information in the given CSV file.
        The CSV file must contain the following column names:
        - 'vox_x','vox_y','vox_z': the voxel coordinates of the
        entry points of each electrode.
        - 'nb_contacts': number of contacts on each electrode
            
        ### Input:
        - path: the path to the CSV file"""
        df = pd.read_csv(path, comment='#')

        # Number of electrodes. Int.
        self.nb_electrodes = len(df)
        # Entry points. Shape (NB_ELECTRODES, 3)
        self.entry_points = df[self._VOX_KEYS].to_numpy(dtype=float)
        # Number of contacts. Shape (NB_ELECTRODES,)
        self.nb_contacts = df['nb_contacts'].to_numpy(dtype=int)