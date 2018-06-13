from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


images_dir = "./JIGSAWS/Suturing/pictures/"    
labels_dir = './JIGSAWS/Suturing/transcriptions/'


LABELS_USECOLS = [0, 1, 2]
LABELS_COL_NAMES = ['start_frame', 'end_frame', 'string_label']
LABELS_CONVERTERS = {2: lambda string_label: int(string_label.replace(b'G', b''))}


user = "B"
trial = 1
trial_name = "Suturing_{}00{}".format(user, trial)

labels_path = labels_dir + trial_name + ".txt"


def get_last_frame(labels_path):
    with open(labels_path,'r') as fh:
        last = fh.readlines()[-1]
    lastLineArray = last.split(" ")
    return int(lastLineArray[1])




def get_trial_name(user, trial):
    """ Form a trial name that matches standard JIGSAWS filenames.
    Args:
        user: A string.
        trial: An integer.
    Returns:
        A string.
    """
    return "Suturing_{}00{}".format(user, trial)



def load_labels(labels_dir, trial_name):
    """ Load kinematics data and labels.
    Args:
        data_dir: A string.
        trial_name: A string.
    Returns:
        A 2-D NumPy array with time on the first axis. Labels are appended
        as a new column to the raw kinematics data (and are therefore
        represented as floats).
    """
    raw_labels_data = np.genfromtxt(labels_path, dtype=np.int,
                                    converters=LABELS_CONVERTERS,
                                    usecols=LABELS_USECOLS)
    #print("rawlabelsdata: ", raw_labels_data)
    
    frames = np.arange(1, get_last_frame(labels_path)+1, dtype=np.int)
    #print("frames: ", frames)
    #print(frames.shape)
    labels = np.zeros(frames.shape, dtype=np.int)
    #print(labels)
    for start, end, label in raw_labels_data:
        mask = (frames >= start) & (frames <= end)
        #print("mask: ",mask)
        labels[mask] = label
        #print("labels[mask]: ",labels[mask])
    
    labels_data = labels.reshape(-1, 1)
    #print(labels_data)
    print("labels: ", labels_data)
    
    return labels_data
    
labels_data = load_labels(labels_dir, trial_name)

