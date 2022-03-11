import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, npy_dir):
        self.root_dir = npy_dir
        self.case_names = [self.root_dir + '/' + x for x in os.listdir(self.root_dir)]
        
        transform_set = [transforms.Lambda(lambda x: x),
                         transforms.RandomRotation(30),
                         transforms.ColorJitter(),
                         transforms.RandomHorizontalFlip(p=1)]
        self.transform = transforms.RandomChoice(transform_set)
        
    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        instance = np.load(self.case_names[index], allow_pickle=True).item()
        x = instance['input'].transpose(2, 0, 1)     # (C, H, W)
        x = torch.from_numpy(x).type(torch.float)    # convert to Tensor to use torchvision.transforms
        x = self.transform(x)
        return x, instance['label']


class EvalDataset(Dataset):
    def __init__(self, npy_dir):
        self.root_dir = npy_dir
        self.case_names = [self.root_dir + '/' + x for x in os.listdir(self.root_dir)]

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        instance = np.load(self.case_names[index], allow_pickle=True).item()
        x = instance['input'].transpose(2, 0, 1)
        x = torch.from_numpy(x).type(torch.float)
        return x, instance['label']