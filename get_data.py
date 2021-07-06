import torch
import albumentations as A
import cv2
import os
import random
import numpy as np
from utils import(
    load_all_from_path,
    np_to_tensor

)

class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, path, device, use_patches=True, resize_to=(400, 400), test_run=False):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to=resize_to
        self.test_run = test_run
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(os.path.join(self.path, 'images'))
        self.y = load_all_from_path(os.path.join(self.path, 'groundtruth'))
        if(self.test_run):
            self.x = self.x[:12]
            self.y = self.y[:12]

        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        
        # I copied the transform from the Albumentations Documentation, -> gives 2% better accuracy from 84% to 86%
        # We have to think which data augmentations make sense here
        
        transform = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose()
            ])
        random.seed(42)

        transformed = transform(image=x, mask=y)
        x = transformed['image']
        y = transformed['mask']

        
        return np_to_tensor(x,self.device), np_to_tensor(y, self.device)

    def __getitem__(self, item):
        return self._preprocess(self.x[item], self.y[[item]])
        # return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))
    
    def __len__(self):
        return self.n_samples