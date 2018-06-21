import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import ActivityDataset
import torchvision
from torch.utils.data import ConcatDataset
from torchvision import models
from torchvision import transforms, utils
from torchvision.transforms import ToPILImage, Resize, RandomHorizontalFlip, RandomRotation, ToTensor, Compose, RandomGrayscale
from PIL import Image
import sys


        
if __name__ == "__main__":
    USER_TO_TRIALS = {
    'B': [1, 2, 3, 4, 5],
    'C': [1, 2, 3, 4, 5],
    'D': [1, 2, 3, 4, 5],
    'E': [1, 2, 3, 4, 5],
    'F': [1, 2, 3, 4, 5],
    'G': [1, 2, 3, 4, 5],
    'H': [1,    3, 4, 5],
    'I': [1, 2, 3, 4, 5]
    }
    #USER_TO_TRIALS = sorted(USER_TO_TRIALS.keys())
    #user = "B"
    #trial = 1
    #trial_name = "Suturing_{}00{}".format(user, trial)
    images_dir = "../JIGSAWS/Suturing/pictures/"
    #trial_dir = trial_name + "_capture1"
    labels_dir = '../JIGSAWS/Suturing/transcriptions/'
    trainErrsTotal = []
    testErrsTotal = []
    trainAccuracyTotal = []
    testAccuracyTotal = []

    def plotErrors( trainErrs, testErrs ):

        trainErrsTotal.append( trainErrs )
        testErrsTotal.append( testErrs )

        plt.clf()
        fig = plt.figure()
        plt.plot( trainErrsTotal, '-', label = "train", color = (0.5,0,0.8) )
        plt.plot( testErrsTotal, '-', label = "test", color = (0.5,0.8,0) )
        #fig.suptitle('test title', fontsize=20)
        plt.xlabel('Epoche', fontsize=14)
        plt.ylabel('Loss', fontsize=14)

        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig( "./errors" )

    def plotAccuracy( trainAccuracy, testAccuracy ):

        trainAccuracyTotal.append( trainAccuracy )
        testAccuracyTotal.append( testAccuracy )
        plt.clf()
        fig = plt.figure()
        plt.plot( trainAccuracyTotal, '-', label = "train", color = (0.5,0,0.8))
        plt.plot( testAccuracyTotal, '-', label = "test", color = (0.5,0.8,0) )

        plt.xlabel('Epoche', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)

        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig( "./accuracy" )
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    img_size = (300, 300)
    composed = Compose([ToPILImage(), Resize(img_size), ToTensor(), normalize])

    dataArray = []
    for user, trials in USER_TO_TRIALS.items():
        if(user == "G"):
            continue
        for trial in trials:
            trial_name = "Suturing_{}00{}".format(user, trial)
            train_dataset = ActivityDataset(trial_name = "Suturing_{}00{}".format(user, trial), 
                                            images_dir = "../JIGSAWS/Suturing/pictures/", 
                                            trial_dir = "Suturing_{}00{}".format(user, trial) + "_capture1",  
                                            labels_dir = '../JIGSAWS/Suturing/transcriptions/',
                                            transform = composed)
            dataArray.append(train_dataset)
    allTraindata = ConcatDataset(([x for x in dataArray]))

    testDataArray = []
    for user, trials in USER_TO_TRIALS.items():
        if(user == "G"):
	        for trial in trials:
	            trial_name = "Suturing_{}00{}".format(user, trial)
	            test_dataset = ActivityDataset(trial_name = "Suturing_{}00{}".format(user, trial), 
	                                            images_dir = "../JIGSAWS/Suturing/pictures/", 
	                                            trial_dir = "Suturing_{}00{}".format(user, trial) + "_capture1",  
	                                            labels_dir = '../JIGSAWS/Suturing/transcriptions/',
	                                            transform = composed)
	            testDataArray.append(test_dataset)

    print(len(allTraindata))

    allTestdata = ConcatDataset(([x for x in testDataArray]))

    train_loader = torch.utils.data.DataLoader(dataset = allTraindata,
                                               batch_size = 51,
                                               shuffle = True,
                                               num_workers = 4
                                               )
    
    test_loader = torch.utils.data.DataLoader(dataset = allTestdata,
                                               batch_size = 51,
                                               shuffle = True,
                                               num_workers = 4
                                               )
    print(len(allTestdata))

    net = models.resnet18(pretrained = True)
    #num_ftrs = net.fc.in_features
    #net.classifier._modules["6"] = nn.Dropout(p = 0.75)
    #net.classifier._modules["7"] = nn.Linear(in_features = 1000, out_features = 11)
    net.fc = nn.Linear(in_features = 8192, out_features = 11)
    net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum = 0.1)

    for epoch in range(0, 1000):
        lossSumm = 0
        lossTestSumm = 0
        print("Epoch {}".format(epoch))
        
        b = 0
        correctTrainSum = 0
        targetSum = 0
        for data in train_loader:
            correct = 0
            b = b + 1
            image, target = data
            #print(target)
            #image, target = Variable(image), Variable(target)
            image = image.cuda()
            target = target.cuda()
            
            output = net(image)
            _, predict = torch.max(output.data, 1)
            predict = predict.type(torch.LongTensor)
            predict = predict.cuda()
            #print(predict[0])
            #print(target[0].data[0])
            for pos in range(0, len(target.data)):
                if(target[pos].data[0] == predict[pos]):
                    correct = correct + 1
            #print(correct)
            #exit()
            criterion = nn.CrossEntropyLoss()
            target = target.squeeze()
            target = target.type(torch.cuda.LongTensor)
            loss = criterion(output, target)
            
            loss.backward()
            lossSumm = lossSumm + loss.data[0]
            correctTrainSum = correctTrainSum + correct
            targetSum = targetSum + target.size(0)
            optimizer.step()
            optimizer.zero_grad()
            #sys.stdout.write("batch: {}".format(b))
            sys.stdout.write("\rbatch:{}, loss:{:.4f}, accuracy:{:.2f}".format(b, loss.data[0], correct / target.size(0) * 100))
                  
            del loss, output, target
        print("aufsummiertes Loss: {:.4f}, accuracy: {:.2f}%".format(lossSumm, correctTrainSum/len(allTraindata)*100))
        
        if(epoch%5 == 0):
            b = 0
            correctTestSum = 0
            for data in test_loader:
	            correct = 0
	            b = b + 1
	            image, target = data
	            #print(target)
	            #image, target = Variable(image), Variable(target)
	            image = image.cuda()
	            target = target.cuda()
	            
	            output = net(image)
	            _, predict = torch.max(output.data, 1)
	            predict = predict.type(torch.LongTensor)
	            predict = predict.cuda()
	            for pos in range(0, len(target.data)):
	                if(target[pos].data[0] == predict[pos]):
	                    correct = correct + 1
	            
	            criterion = nn.CrossEntropyLoss()
	            target = target.squeeze()
	            target = target.type(torch.cuda.LongTensor)
	            loss = criterion(output, target)

	            lossTestSumm = lossTestSumm + loss.data[0]

	            correctTestSum = correctTestSum + correct
	            print("batch:{}, accuracy:{:.2f}%".format(b, correct / target.size(0) * 100))
            plotErrors(lossSumm/len(train_loader.dataset), lossTestSumm/len(test_loader.dataset))
            plotAccuracy(correctTrainSum/ len(allTraindata)*100, correctTestSum/ len(allTestdata)* 100)

        if(epoch%100 == 0):
            print("saved")
            torch.save(net.state_dict(), "./model{}.pth".format(epoch))
        sys.stdout.write("")
        
        torch.cuda.empty_cache()