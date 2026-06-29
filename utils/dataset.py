import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import torchvision
from torchvision.transforms import functional as F
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as L

class Dataset(Dataset):
    # annotations_file: csv with data input information (filepath, label and patient)
    # img_dir: directory with images 
    # transform: resizes and applies data augmentations if necessary to the input of the network
    # mean: mean of the images of the whole dataset by channels (if [0,0,0], no standardization applied)
    # std: std of the images of the whole dataset by channels (if [1,1,1], no standardization applied)

    def __init__(self, annotations_file, img_dir, transform, mean, std):  
        self.csv = annotations_file
        self.transform = transform
        self.img_dir = img_dir
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        img_path = self.csv["Path"].iloc[idx]
        # print(os.path.join(self.img_dir, img_path))
        image = Image.open(os.path.join(self.img_dir, img_path))    
        img_array = np.array(image)
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
            image = Image.fromarray(img_array)
        label = self.csv["Label"].iloc[idx]

        if self.transform:
            image = self.transform(image)
        return image, label, img_path



def get_mean_std(train_csv, img_dir):
    r_means, g_means, b_means, r_stds, g_stds, b_stds = [[] for i in range(6)]

    for line in train_csv["Path"]: 
        image = Image.open(os.path.join(img_dir, line)) 
        image = transforms.ToTensor()(image)   
        img_array = np.array(image)
        r_channel = img_array[:, :, 0]
        g_channel = img_array[:, :, 1]
        b_channel = img_array[:, :, 2]
        r_means.append(r_channel.mean())
        g_means.append(g_channel.mean())
        b_means.append(b_channel.mean())
        r_stds.append(r_channel.std())
        g_stds.append(g_channel.std())
        b_stds.append(b_channel.std())
    mean_r = np.mean(r_means)
    mean_g = np.mean(g_means)
    mean_b = np.mean(b_means)
    std_r = np.std(r_means)
    std_g = np.std(g_means)
    std_b = np.std(b_means)

    return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]