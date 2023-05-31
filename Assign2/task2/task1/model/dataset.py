import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class CIFAR_100_Dataset(Dataset):
    def __init__(self, data, label, augment=None, **kwargs):
        self.data = torch.tensor(data)
        self.label = torch.tensor(label)
        self._label_num = max(label) + 1
        self.augment = augment
        self._check_param()

        if self.augment is not None:
            self.prob = kwargs['prob'] # whether to augmentation
            self._beta = kwargs['beta']
            if self.augment == 'mixup':
                self.augment = self._mixup_aug
            elif self.augment == 'cutout':
                self.augment = self._cutout_aug
            else:
                self.augment = self._cutmix_aug
    
    def _check_param(self):
        assert len(self.data) == len(self.label)
        if self.augment is not None:
            self.augment = self.augment.lower()
            if self.augment not in ['cutmix', 'cutout', 'mixup']:
                raise NotImplementedError()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            prob = np.random.uniform()
            img = self.data[idx]
            label = F.one_hot(self.label[idx], self._label_num)
            if self.augment is not None:
                if prob < self.prob:
                    img, label = self.augment(img, label)
            
            return img, label
        elif isinstance(idx, slice) or isinstance(idx, list):
            img = self.data[idx]
            label = F.one_hot(self.label[idx], self._label_num)
            return img, label
        else:
            raise ValueError()
    
    def __len__(self):
        return len(self.data)
    
    def _mixup_aug(self, img, label):
        lambda_val = np.random.beta(self._beta, self._beta)
        aug_idx = np.random.randint(len(self.data))
        aug_img = self.data[aug_idx]
        aug_label = self.label[aug_idx]

        aug_label = F.one_hot(aug_label, self._label_num)

        return_img, return_label = None, None
        return_img = lambda_val * img + (1 - lambda_val) * aug_img
        return_label = lambda_val * label + (1 - lambda_val) * aug_label
        return return_img, return_label    

    def _cutmix_aug(self, img, label):
        lambda_val = np.random.beta(self._beta, self._beta)
        aug_idx = np.random.randint(len(self.data))
        aug_img = self.data[aug_idx]
        aug_label = self.label[aug_idx]

        aug_label = F.one_hot(aug_label, self._label_num)

        img_height = img.shape[0]
        img_width = img.shape[1]
        
        cut_rate = np.sqrt(1 - lambda_val)
        cut_height = int(img_height * cut_rate)
        cut_width = int(img_width * cut_rate)
        height_idx = np.random.randint(img_height)
        width_idx = np.random.randint(img_width)
        reindex_cut_height_lower = np.clip(height_idx - cut_height // 2, 0, img_height)
        reindex_cut_height_upper = np.clip(height_idx + cut_height // 2, 0, img_height)
        reindex_cut_width_lower = np.clip(width_idx - cut_width // 2, 0, img_width)
        reindex_cut_width_upper = np.clip(width_idx + cut_width // 2, 0, img_width)
        
        cut_percent = ((reindex_cut_height_upper - reindex_cut_height_lower) * (reindex_cut_width_upper - reindex_cut_width_lower)) / (img_height * img_width)
        return_img, return_label = img.clone(), label.clone()
        return_img[reindex_cut_height_lower: reindex_cut_height_upper, reindex_cut_width_lower: reindex_cut_width_upper, :] = aug_img[reindex_cut_height_lower: reindex_cut_height_upper, reindex_cut_width_lower: reindex_cut_width_upper, :]
        return_label = (1 - cut_percent) * label + cut_percent * aug_label
        return return_img, return_label

    def _cutout_aug(self, img, label):
        lambda_val = np.random.beta(self._beta, self._beta)
        img_height = img.shape[0]
        img_width = img.shape[1]
        cut_rate = np.sqrt(1 - lambda_val)
        cut_height = int(img_height * cut_rate)
        cut_width = int(img_width * cut_rate)
        height_idx = np.random.randint(img_height)
        width_idx = np.random.randint(img_width)
        reindex_cut_height_lower = np.clip(height_idx - cut_height // 2, 0, img_height)
        reindex_cut_height_upper = np.clip(height_idx + cut_height // 2, 0, img_height)
        reindex_cut_width_lower = np.clip(width_idx - cut_width // 2, 0, img_width)
        reindex_cut_width_upper = np.clip(width_idx + cut_width // 2, 0, img_width)

        return_img = img.clone()
        return_img[reindex_cut_height_lower: reindex_cut_height_upper, reindex_cut_width_lower: reindex_cut_width_upper, :] = 0
        return return_img, label