import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
import matplotlib
matplotlib.use('Agg')
import Net
import matplotlib.pyplot as plt
import dataloader
from dataloader import KinematicDataset
import torchvision
from torch.utils.data import ConcatDataset
from torchvision import models
from torchvision import transforms, utils
from torchvision.transforms import ToPILImage, Resize, RandomHorizontalFlip, RandomRotation, ToTensor, Compose, RandomGrayscale
from PIL import Image
import sys

with open("../Suturing/standardized_data.pkl", 'rb') as f:
    pkldict = pickle.load(f)

datadict = pkldict["all_data"]
print(pkldict["all_data"])
all_trial_names = pkldict["all_trial_names"] #array
user_to_trial_names = pkldict["user_to_trial_names"] #dict
all_users = pkldict["all_users"] #array



composed = Compose(ToTensor())
dataArray = []


for user, trials in user_to_trial_names.items():
    if(user == "B"):
        for trial in trials:
            print(trial)
            train_dataset = KinematicDataset(trial_name = trial, 
                                             datadict = datadict, 
                                             transform = composed
                                             )
            print(len(train_dataset))
            dataArray.append(train_dataset)
print(dataArray)
allTraindata = ConcatDataset(([x for x in dataArray]))

print(len(allTraindata))


train_loader = torch.utils.data.DataLoader(dataset = allTraindata,
                                               batch_size = 512,
                                               shuffle = False,
                                               num_workers = 4
                                               )

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net.LSTM(num_class, lstm_size, dropout)
net.cuda()

epochs = 10
for epoch in range(epochs):
    print("Epoch: {}".format(epoch))
    lossSumm = 0
    lossTestSumm = 0
    batch_nr = 0
    correctTrainSum = 0
    correctTestSum = 0
    targetSum = 0
    hidden_state = net.init_hidden(batch_size = batch_size, device = device)

    model.train()
    for data in train_loader:
        correct = 0
        batch_nr += 1
        kinematics, label = data

        kinematics = kinematics.cuda()
        label = label.cuda()

        output, hidden_state = net(kinematics, hidden_state)
        _, predict = torch.max(output.data, 1)
        predict = predict.type(torch.LongTensor)
        predict = predict.cuda()
        
        for pos in range(0, len(target.data)):
            if(label[pos].data[0] == predict[pos]):
                correct = correct + 1
        target = target.squeeze()
        target = target.type(torch.cuda.LongTensor)
        loss = criterion(output, target)

        loss.backward()
        lossSumm = lossSumm + loss.item()
        correctTrainSum = correctTrainSum + correct
        targetSum = targetSum + target.size(0)
        optimizer.step()
        optimizer.zero_grad()

        h, c = hidden_state
        h = h.detach()
        c = c.detach()
        hidden_state = (h, c)

        sys.stdout.write("\rbatch:{}, loss:{:.4f}, accuracy:{:.2f}".format(batch_nr, loss.item(), correct / target.size(0) * 100))
         
        del loss, output, target

    print('\n', end = ' ', flush=True)

    print("aufsummiertes Loss: {:.4f}, accuracy: {:.2f}%".format(lossSumm, correctTrainSum/len(allTraindata)*100))
    torch.cuda.empty_cache()
    batch_nr = 0
