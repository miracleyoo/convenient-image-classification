# coding: utf-8
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class TRAIN_LOADER(Dataset):
    def __init__(self, data, opt, transform=None):
        super(TRAIN_LOADER, self).__init__()
        self.data = data
        self.opt = opt
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path, label = self.data[index]
        label = np.array(label)
        label = np.eye(self.opt.NUM_CLASSES)[label]
        img = np.array(Image.open(data_path))
        if img.shape[2] == 1:
            img = np.append(img, img, 0)
            img = np.append(img, img, 0)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        if self.transform:
            sample = self.transform(img)
        else:
            sample = img
        return sample, label.astype(np.float32), str(data_path)

class PRED_LOADER(Dataset):
    def __init__(self, data, transform=None):
        super(PRED_LOADER, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path = self.data[index]
        img = np.array(Image.open(data_path))
        if img.shape[2] == 1:
            img = np.append(img, img, 0)
            img = np.append(img, img, 0)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        if self.transform:
            sample = self.transform(img)
        else:
            sample = img
        return sample, str(data_path)