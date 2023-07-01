import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class Animals10(Dataset):

    def __init__(self, root, train, transform, target_map, train_val=[1, 0], color=False, rotation=False) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.target_map = target_map
        self.train_val = train_val
        self.file = pd.read_csv(root, index_col=0)
        self.color = color
        self.rotation = rotation

        self.train_idx = int(train_val[0]/sum(train_val)*self.file.shape[0])

        if self.train:
            self.path_file = self.file.iloc[:self.train_idx, :]
        else:
            self.path_file = self.file.iloc[self.train_idx:, :]

    def __getitem__(self, index):
        raw_class_i, path_i = self.file.iloc[index, :]
        class_i = self.target_map[raw_class_i]
        img_i = cv2.imread(path_i, 1)

        if self.color:
            img_lab = cv2.cvtColor(img_i, cv2.COLOR_BGR2LAB)
            img_lab = self.transform(img_lab)
            return img_lab[[0, 0, 0], :, :], img_lab[1:, :, :]

        if self.rotation:
            rand_idx = np.random.choice([0, 1, 2, 3])
            img_rotation = np.rot90(img_i, rand_idx)
            return self.transform(img_rotation), rand_idx

        img_i = self.transform(img_i)
        return img_i, class_i

    def __len__(self):
        return self.path_file.shape[0]


class Rotation(nn.Module):
    def __init__(self, input_channel, input_shape, cls_num) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.input_shape = input_shape
        self.cls_num = cls_num
        self.fc = nn.Linear(self.input_channel, self.cls_num)

    def forward(self, x):
        out = F.avg_pool2d(x, self.input_shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Color(nn.Module):

    def __init__(self, input_channel=512, output_channel=2) -> None:
        super().__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.convT1 = nn.ConvTranspose2d(
            self.input_channel, 32, 2, stride=4, padding=0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.convT2 = nn.ConvTranspose2d(
            32, self.output_channel, 2, stride=4, padding=3)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=1,
                               stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.convT1(x)
        out = self.conv1(out)
        out = self.convT2(out)
        out = self.conv2(out)
        return out


class clf_head(nn.Module):
    def __init__(self, input_channel, input_shape, cls_num) -> None:
        super().__init__()

        self.input_channel = input_channel
        self.input_shape = input_shape
        self.cls_num = cls_num
        self.fc = nn.Linear(self.input_channel, self.cls_num)

    def forward(self, x):
        out = F.avg_pool2d(x, self.input_shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
