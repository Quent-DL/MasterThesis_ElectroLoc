import pandas as pd
import numpy as np
from typing import Optional, Self


class DataFrameContacts(pd.DataFrame):
    _VOX_KEYS = ['vox_x', 'vox_y', 'vox_z']
    _WORLD_KEYS = ['world_x', 'world_y', 'world_z']
    _LABEL_KEY = 'electrode_id'
    _CID_KEY = 'c_id'
    # Only for ground truths
    _TAG_DCC_KEY = 'tag_dcc'

    def __init__(self,
                 vox_coords: Optional[np.ndarray] = None,
                 world_coords: Optional[np.ndarray] = None,
                 labels: Optional[np.ndarray] = None,
                 positional_ids: Optional[np.ndarray] = None,
                 ):
        super().__init__()

        if vox_coords is not None:
            self.set_vox_coordinates(vox_coords)
        if world_coords is not None:
            self.set_world_coordinates(world_coords)
        if labels is not None:
            self.set_labels(labels)
        if positional_ids is not None:
            self.set_positional_ids(positional_ids)

    @staticmethod
    def from_csv(path: str) -> Self:
        """TODO write documentation"""
        all_valid_keys = (
            DataFrameContacts._VOX_KEYS 
            + DataFrameContacts._WORLD_KEYS 
            + [DataFrameContacts._LABEL_KEY]
            + [DataFrameContacts._CID_KEY]
            + [DataFrameContacts._TAG_DCC_KEY])

        # New instance with a copy of the relevant content
        instance = DataFrameContacts()

        df = pd.read_csv(path, comment='#')
        for key in df.keys():
            if key in all_valid_keys:
                instance[key] = df[key]
        
        return instance
            
    def set_vox_coordinates(self, coords: np.ndarray) -> None:
        """TODO write documentation"""
        for key, values in zip(self._VOX_KEYS, coords.T):
            self[key] = values

    def get_vox_coordinates(self) -> np.ndarray:
        """TODO write documentation"""
        return self[self._VOX_KEYS].to_numpy(dtype=float)

    def set_world_coordinates(self, coords: np.ndarray) -> None:
        """TODO write documentation"""
        for key, values in zip(self._WORLD_KEYS, coords.T):
            self[key] = values

    def get_world_coordinates(self) -> np.ndarray:
        """TODO write documentation"""
        return self[self._WORLD_KEYS].to_numpy(dtype=float)

    def set_labels(self, labels: np.ndarray) -> None:
        """TODO write documentation"""
        # Checking that there are ways to identify the contacts
        _nb_vox, _nb_world = 0, 0 
        for key in self._VOX_KEYS:
            if key in self:
                _nb_vox += 1
        for key in self._WORLD_KEYS:
            if key in self:
                _nb_world += 1

        if _nb_vox == 3 or _nb_world == 3:
                self[self._LABEL_KEY] = labels
                self.sort_values(
                    by=[self._LABEL_KEY], 
                    axis='index', inplace=True)
        else: 
            raise RuntimeError("Labels cannot be set before any type of coordinates.")

    def get_labels(self) -> np.ndarray:
        """TODO write documentation"""
        dtype = self[self._LABEL_KEY].dtype
        return self[self._LABEL_KEY].to_numpy(dtype)
    
    def set_positional_ids(self, ids: np.ndarray) -> None:
        """TODO write documentation"""
        if self._LABEL_KEY in self:
            self[self._CID_KEY] = ids
            self.sort_values(
                by=[self._LABEL_KEY, self._CID_KEY], 
                axis='index', inplace=True)
        else:
            raise RuntimeError("Positional ids cannot be set before electrode ids (labels).")

    def get_positional_ids(self) -> np.ndarray:
        """TODO write documentation"""
        return self[self._CID_KEY].to_numpy(dtype=int)