import logging
from pathlib import Path

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from scipy.io import loadmat
from config import DefaultConfig

from utils.guiding_signals import PointGuidingSignal



class NuclickDataset(Dataset):
    '''Dataset class for NuClick
    
    This class includes all the processes needed for loading patches previously
    extracted for NuClick model training (using `data.patch_extraction` module).
    '''

    def __init__(self, patch_dir: str, phase: str = 'train', scale: float = None, drop_rate: float = 0, jitter_range: int = 3):
        self.patch_dir = Path(patch_dir)
        if phase.lower() not in {'train', 'validation', 'val', 'test'}:
            raise ValueError(f'Invalid running phase of: {patch_dir}. Phase should be `"train"`, `"validation"`, or `"test"`.')
        self.phase = phase.lower()
        self.scale = scale
        self.drop_rate = drop_rate
        self.jitter_range = jitter_range

        self.file_paths = list(self.patch_dir.glob('*.mat'))
        if len(self.file_paths)==0:
            raise RuntimeError(f'No input file found in {patch_dir}, make sure you put your images there')
        self.ids = [path.stem for path in self.file_paths]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    
    def __len__(self):
        return len(self.ids)


    @classmethod
    def preprocess(slef, img, scale, is_mask):
        [h, w] = img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        if is_mask:
            # Convert mask to black-and-white:
            img_ndarray = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_NEAREST)
            return img_ndarray
        else:
            img_ndarray = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_CUBIC)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray
       

    def __getitem__(self, idx):
        mat_file = loadmat(self.file_paths[idx])
        img = mat_file['img']
        mask = mat_file['mask']
        others = mat_file['others']

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        others = self.preprocess(others, self.scale, is_mask=True)

        # create the guiding signal generator
        signal_gen = PointGuidingSignal(mask, others, perturb=DefaultConfig.perturb)
        inc_signal = signal_gen.inclusion_map()
        exc_signal = signal_gen.exclusion_map(random_drop=self.drop_rate, random_jitter=self.jitter_range)

        input = np.concatenate((img, inc_signal[np.newaxis, ...], exc_signal[np.newaxis, ...]), axis=0)

        return {
            'image': torch.as_tensor(input.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
        }
