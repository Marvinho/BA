from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import preDataProcessing

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")




class ActivityDataset(Dataset):
    """Face Landmarks dataset."""
    #user = "B"
    #trial = 1
    #trial_name = "Suturing_{}00{}".format(user, trial)
    images_dir = "../JIGSAWS/Suturing/pictures/"
    #trial_dir = trial_name + "_capture1"
    labels_dir = '../JIGSAWS/Suturing/transcriptions/'
    

    def __init__(self, images_dir, trial_name, trial_dir, labels_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = preDataProcessing.load_labels(labels_dir, trial_name)
        self.images_dir = images_dir
        self.transform = transform
        self.trial_dir = trial_dir
        self.trial_name = trial_name
        self.labels_dir = labels_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir,
                                self.trial_dir, self.trial_name + "_capture1_" +str(idx+1).zfill(4)+".png")
        #print(img_name)
        image = io.imread(img_name)
        landmarks = self.landmarks_frame[idx]-1
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            image = self.transform(image)

        return image, landmarks

#FaceLandmarksDataset(Dataset)
"""
if __name__ == "__main__":
    user = "B"
    trial = 1
    trial_name = "Suturing_{}00{}".format(user, trial)
    images_dir = "./JIGSAWS/Suturing/pictures/"
    trial_dir = trial_name + "_capture1_30frames"
    labels_dir = './JIGSAWS/Suturing/transcriptions/'
    train_dataset = ActivityDataset(user = "B",trial = 1, 
                                        trial_name = "Suturing_{}00{}".format(user, trial), 
                                        images_dir = "./JIGSAWS/Suturing/pictures/", 
                                        trial_dir = trial_name + "_capture1", 
                                        labels_dir = './JIGSAWS/Suturing/transcriptions/')
    
    for i in range(920):
        x, _ = train_dataset[i]
        print(i)
        _, y = train_dataset[i]
        print(i, x.shape, y)
        #print(x)
        #print(labels)
"""

    
