from __future__ import division, print_function

import os
import time
import argparse
import shutil
import pickle
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib

#import matplotlib.pyplot as plt

import pre_data
import data
import warnings
warnings.filterwarnings("ignore")
   

class KinematicDataset(Dataset):
    
    def __init__(self,trial_name, datadict, transform = None):
        self.datadict = datadict
        self.transform = transform
        self.trial_name = trial_name

    def __len__(self):

        return len(self.datadict[self.trial_name])

    def __getitem__(self, idx):
        kinematics = datadict[trial_name][idx][:-1]
        label = datadict[trial_name][idx][-1]
        sample = {"kinematic": kinematics, "label": label}
        return kinematics, label


