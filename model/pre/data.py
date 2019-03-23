import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image


class MainDataset(Dataset):
    def __init__(self, x_dataset, y_dataset, x_tfms):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.x_tfms = x_tfms

    def __len__(self):
        return self.x_dataset.__len__()

    def __getitem__(self, index):
        x = self.x_dataset[index]
        y = self.y_dataset[index]
        if self.x_tfms is not None:
            x = self.x_tfms(x)
        return x, y


class ImageDataset(Dataset):
    def __init__(self, paths_to_imgs):
        self.paths_to_imgs = paths_to_imgs

    def __len__(self):
        return len(self.paths_to_imgs)

    def __getitem__(self, index):
        img = Image.open(self.paths_to_imgs[index])
        return img


class LabelDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]


class DataFrameDataset(Dataset):
    def __init__(self, df_data, data_dir='./', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name, label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name+'.tif')
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


trans_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(64, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

trans_valid = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(64, padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

trans_test = transforms.Compose([
    transforms.Resize((196, 196)),
    transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(saturation=0.5),
        transforms.ColorJitter(hue=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    ]),
    transforms.RandomChoice([
        transforms.RandomRotation((0, 0)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270)),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation((90, 90)),
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation((270, 270)),
        ])
    ]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
