import torch
import torch.nn as nn
from albumentations import Compose as ACompose
from albumentations import HorizontalFlip, ShiftScaleRotate, CoarseDropout
from albumentations import Normalize as ANormalize
from albumentations.pytorch import ToTensorV2
import numpy as np

# Data Augmentation
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# Albumentations works on uint8 images pre-normalization; use 0–255 fill_value
mean_255 = tuple(int(m*255) for m in mean)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
                                   padding=padding, groups=in_channels, dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def albumentations_transform():
    return ACompose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                      fill_value=mean_255, mask_fill_value=None),
        ANormalize(mean=mean, std=std),
        ToTensorV2()
    ])

class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, label

# Model Architecture
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        
        # C1 block - Initial features extraction
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(3, 64, kernel_size=5, padding=2),  # RF: 5x5
            nn.ReLU(),
            nn.BatchNorm2d(64),
            DepthwiseSeparableConv(64, 96, kernel_size=3, padding=1),  # RF: 7x7
            nn.ReLU(),
            nn.BatchNorm2d(96),
        )

        # C2 block - Dilated convolution for expanded RF
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(96, 128, kernel_size=3, padding=2, dilation=2),  # RF: 15x15
            nn.ReLU(),
            nn.BatchNorm2d(128),
            DepthwiseSeparableConv(128, 160, kernel_size=3, padding=1),  # RF: 19x19
            nn.ReLU(),
            nn.BatchNorm2d(160),
        )

        # C3 block - Further feature processing
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(160, 192, kernel_size=3, padding=4, dilation=4),  # RF: 35x35
            nn.ReLU(),
            nn.BatchNorm2d(192),
            DepthwiseSeparableConv(192, 224, kernel_size=3, padding=1),  # RF: 39x39
            nn.ReLU(),
            nn.BatchNorm2d(224),
        )

        # C4 block - Final convolution with stride 2
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(224, 256, kernel_size=3, stride=2, padding=2, dilation=2),  # RF: 47x47
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

