import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from scipy.io import loadmat
from config import DefaultConfig

from utils.guiding_signals import PointGuidingSignal


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    
    def __len__(self):
        return len(self.ids)


    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if is_mask:
            # Convert mask to black-and-white:
            img_ndarray = img_ndarray > 0
            return img_ndarray

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray


    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)


    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

    
class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')

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
