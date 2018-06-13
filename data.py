from __future__ import print_function, division
import os
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from os.path import join
from PIL import Image
import numpy as np
import pickle








LABELS_USECOLS = [0, 1, 2]
LABELS_COL_NAMES = ['start_frame', 'end_frame', 'string_label']
LABELS_CONVERTERS = {2: lambda string_label: int(string_label.replace(b'G', b''))}


KINEMATICS_USECOLS = [c-1 for c in [39, 40, 41, 51, 52, 53, 57,
                                    58, 59, 60, 70, 71, 72, 76]]
KINEMATICS_COL_NAMES = ['pos_x', 'pos_y', 'pos_z', 'vel_x',
'vel_y', 'vel_z', 'gripper']*2


images_dir = "./JIGSAWS/Suturing/pictures/"    


labels_dir = './JIGSAWS/Suturing/transcriptions/'
user = "B"
trial = 1
trial_name = "Suturing_{}00{}".format(user, trial)
print(trial_name)
labels_path = labels_dir + trial_name + ".txt"
#import random
#random.shuffle(images)
#print(images)
class_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
classes = ['G%d' % id for id in class_ids]
print(classes)
#print(class_to_idx)
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose, Resize

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_size = (224, 224)
my_transform = Compose([Resize(image_size), ToTensor()])


traindata = ImageFolder(root=images_dir, transform=my_transform)
print(traindata)
trainloader = DataLoader(traindata, batch_size = 4, shuffle = True)
print(trainloader)

#############################################################################################
def load_kinematics(data_dir, trial_name):
    """ Load kinematics data.
    Args:
        data_dir: A string.
        trial_name: A string.
    Returns:
        A 2-D NumPy array with time on the first axis.
    """

    #kinematics_dir = os.path.join(data_dir, 'kinematics', 'AllGestures')
    kinematics_path = os.path.join("./JIGSAWS/Suturing/kinematics/AllGestures/", trial_name + ".txt")
    data = np.loadtxt(kinematics_path, dtype=np.float,
                      usecols=KINEMATICS_USECOLS)
    print(data)
    print(data.shape)
    print(data.shape[0])
    return data
load_kinematics(data_dir = "./JIGSAWS/Suturing/",trial_name = trial_name)

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
    kinematics_data = load_kinematics(data_dir = "./JIGSAWS/Suturing/",trial_name = trial_name)
    raw_labels_data = np.genfromtxt(labels_path, dtype=np.int,
                                    converters=LABELS_CONVERTERS,
                                    usecols=LABELS_USECOLS)
    print("rawlabelsdata: ", raw_labels_data)
    print("kinematics data shaep", kinematics_data.shape[0])
    frames = np.arange(start = 1,stop = kinematics_data.shape[0]+1, dtype=np.int)
    print("frames: ", frames)
    #print(frames.shape)
    labels = np.zeros(frames.shape, dtype=np.int)
    print(labels)
    for start, end, label in raw_labels_data:
        mask = (frames >= start) & (frames <= end)
        #print("mask: ",mask)
        labels[mask] = label
        #print("labels[mask]: ",labels[mask])
    
    labels_data = labels.reshape(-1, 1)
    print(labels_data)
    print("labelsdatashape", labels_data.shape)
    
    return raw_labels_data
    
labels_data = load_labels(labels_dir, trial_name)
###############################################################################################










#testdata = ImageFolder(root=test_dir, transform=my_transform)
#testloader = DataLoader(testdata, batch_size = 4, shuffle = True)

print(iter(trainloader))
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(labels)
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    #print(labels)
# functions to show an image
"""
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#print(iter(trainloader))
# get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#print(labels)
# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
"""