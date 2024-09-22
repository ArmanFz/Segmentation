
import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, images_directory, masks_directory, transform=None):
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.images_filenames = os.listdir(images_directory)
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = Image.open(os.path.join(self.images_directory, image_filename))
        mask = Image.open(os.path.join(self.masks_directory, image_filename))
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        numerator = 2 * torch.sum(pred * target)
        denominator = torch.sum(pred + target)
        dice_score = (numerator) / (denominator + self.eps)
        return 1 - dice_score