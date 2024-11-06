import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
import random
from PIL import Image
import cv2
import numpy as np
import os


class C101_Dataset(Dataset):
    def __init__(self, transform, folder_path):
        self.folder_path = folder_path
        self.transform = transform
        self.image_list = os.listdir(folder_path)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.folder_path, img_name)
        
        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = Image.fromarray(image)

        # Perform any additional transformations on the image if needed (e.g., normalization)
        image = self.transform(image)
        
        # Convert image to tensor
        image = image.clone().detach()
        
        return image, img_name