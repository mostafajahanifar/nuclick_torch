import logging
from pathlib import Path

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from scipy.io import loadmat
from config import DefaultConfig
import albumentations as alb
from albumentations.pytorch import ToTensorV2

from utils.guiding_signals import PointGuidingSignal


class NuclickDataset(Dataset):
    '''Dataset class for NuClick
    
    This class includes all the processes needed for loading patches previously
    extracted for NuClick model training (using `data.patch_extraction` module).
    '''

    def __init__(self, patch_dir: str, phase: str = 'train', scale: float = None, drop_rate: float = 0, jitter_range: int = 3, object_weights=None, augment=True):
        self.patch_dir = Path(patch_dir)
        if phase.lower() not in {'train', 'validation', 'val', 'test'}:
            raise ValueError(f'Invalid running phase of: {patch_dir}. Phase should be `"train"`, `"validation"`, or `"test"`.')
        self.phase = phase.lower()
        self.scale = scale
        self.drop_rate = drop_rate
        self.jitter_range = jitter_range
        self.object_weights = object_weights
        self.augment = augment

        self.file_paths = list(self.patch_dir.glob('*.mat'))
        if len(self.file_paths)==0:
            raise RuntimeError(f'No input file found in {patch_dir}, make sure you put your images there')
        self.ids = [path.stem for path in self.file_paths]
        
        # creating the augmentation
        stain_matrix = np.array([[0.91633014, -0.20408072, -0.34451435],
                                 [0.17669817, 0.92528011, 0.33561059]])
        self.train_augs = alb.Compose([
            alb.OneOf([
                alb.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=(-30,20), val_shift_limit=0, always_apply=False, p=0.75),#.8
                alb.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.75), #.7
                ],
                p=1.),
            alb.OneOf([
                alb.GaussianBlur(blur_limit=(3, 5), p=0.5),
                alb.Sharpen(alpha=(0.1, 0.3), lightness=(1.0, 1.0), p=0.5),
                alb.ImageCompression (quality_lower=30, quality_upper=80, p=0.5),
                ],
                p=1.),
            alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=180, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            alb.Flip(p=0.5)
            ],
            additional_targets={'others':'mask'},
            p=0.5)

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    
    def __len__(self):
        return len(self.ids)


    @classmethod
    def img_scale(slef, img, scale, is_mask):   
        [h, w] = img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        if is_mask:
            img_ndarray = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_NEAREST)
        else:
            img_ndarray = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_CUBIC)

        return img_ndarray

    @staticmethod
    def pre_processing(img, inc_signal, exc_signal):
        # Normalize the input image
        img = np.float32(img)/255.
        img = np.moveaxis(img, -1, 0)
        input = np.concatenate((img, inc_signal[np.newaxis, ...], exc_signal[np.newaxis, ...]), axis=0)
        return input

    @staticmethod
    def pad_to_shape(img, shape, is_mask):
        img_shape = img.shape[:2]
        shape_diff = np.array(shape) - np.array(img_shape)
        if is_mask:
            img_padded = np.pad(img, [(0, shape_diff[0]), (0, shape_diff[1])], mode='constant', constant_values=0)
        else:
            img_padded = np.pad(img, [(0, shape_diff[0]), (0, shape_diff[1]), (0,0)], mode='constant', constant_values=0)
        return img_padded
    
    def __getitem__(self, idx):
        mat_file = loadmat(self.file_paths[idx])
        img = mat_file['img']
        mask = mat_file['mask']
        others = mat_file['others']
        
        ## correct for patches smaller than 128x128
        # TODO: fix this in patch extraction
        if img.shape[0]<DefaultConfig.patch_size or img.shape[1]<DefaultConfig.patch_size:
            desired_shape = [DefaultConfig.patch_size, DefaultConfig.patch_size]
            img = self.pad_to_shape(img, desired_shape, is_mask=False)
            mask = self.pad_to_shape(mask, desired_shape, is_mask=True)
            others = self.pad_to_shape(others, desired_shape, is_mask=True)
        
        # scale the image, mask, and others
        if self.scale is not None and self.scale != 1:
             img = self.img_scale(img, self.scale, is_mask=False)
             mask = self.img_scale(mask, self.scale, is_mask=True)
             others = self.img_scale(others, self.scale, is_mask=True)

        # image and mask augmentation during the training
        if self.phase == 'train' and self.augment:
            augmented_data = self.train_augs(image=img, mask=mask, others=others)
            img = augmented_data["image"]
            mask = augmented_data["mask"]
            others = augmented_data["others"]

        # create the guiding signals
        signal_gen = PointGuidingSignal(mask, others, perturb=DefaultConfig.perturb)
        inc_signal = signal_gen.inclusion_map()
        exc_signal = signal_gen.exclusion_map(random_drop=self.drop_rate, random_jitter=self.jitter_range)

        # Create input and weight maps
        input = self.pre_processing(img, inc_signal, exc_signal)
        if self.object_weights is not None:
            if len(self.object_weights)==2:
                weight_map = 1 + self.object_weights[0]*np.float32(mask) + self.object_weights[1]*np.float32(others>0)
                weights =  torch.as_tensor(weight_map[np.newaxis, ...].copy()).long().contiguous()
            else:
                raise ValueError('object_weights should be a list or tuple in this format: '
                                 '(Desired_Object_Weight, Other_Objects_Weight)')
        else:
            weights = None

        return {'image': torch.as_tensor(input.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask[np.newaxis, ...].copy()).long().contiguous(),
                'weights': weights
                }
