import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.serialization import load_lua
from collections import defaultdict


import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models
import torchfile
from torch.utils.data import TensorDataset

#from torchsample.transforms import Compose, RangeNorm, ToTensor, TypeCast, RandomGamma, RandomBrightness, RandomSaturation, ChannelsFirst, ChannelsLast


import time
import copy
import os

import numpy as np

#Set Cuda Device
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#Root location for the CIFAR-100 dataset
root = "./cs231n"

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN = 49000
NUM_VAL = 1000

#train_filename = '/Users/peterjalbert/Desktop/RUDSSP/cifar100_cnn/cs231n/cifar-100-python/train'
train_filename = './cs231n/cifar-100-python/train'
#test_filename = '/Users/peterjalbert/Desktop/RUDSSP/cifar100_cnn/cs231n/cifar-100-python/test'
test_filename = './cs231n/cifar-100-python/test'
train_filedict = unpickle(train_filename)
test_filedict = unpickle(test_filename)
#print (filedict.keys())
#print(filedict['fine_labels'])
train_coarse_list = train_filedict[b'coarse_labels']
test_coarse_list = test_filedict[b'coarse_labels']
train_fine_list = train_filedict[b'fine_labels']
test_fine_list = test_filedict[b'fine_labels']


############## Load Train Data ################
cifar100_train = dset.CIFAR100(root, train=True, download=True,
                          transform=T.ToTensor())

cifar100_train.train_labels = train_coarse_list


#Pre-process by subtracting the overall mean
reshaped = np.reshape(cifar100_train.train_data, (50000,3072))
np.subtract(reshaped, np.reshape(np.mean(reshaped, axis=1), (50000, 1)))
cifar100_train.train_data = np.reshape(reshaped, (50000, 32, 32, 3))


loader_train = DataLoader(cifar100_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))


########### Load Validation Set ################

cifar100_val = dset.CIFAR100(root, train=True, download=True,
                           transform=T.ToTensor())

cifar100_val.train_labels = train_coarse_list

loader_val = DataLoader(cifar100_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))


############## Load Test Data ####################

cifar100_test = dset.CIFAR100(root, train=False, download=True,
                          transform=T.ToTensor())

cifar100_test.test_labels = test_coarse_list

loader_test = DataLoader(cifar100_test, batch_size=64)


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class LeNet(nn.Module):
    """
    For testing purposes. Poor performance.
    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 20, 5)
        self.flatten = Flatten()
        self.fc1   = nn.Linear(20*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 20)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2, stride=2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2, stride=2)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class DecisionBlock(nn.Module):
    def __init__(self):
        super(DecisionBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(256, 512, 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace = True)

        self.conv6 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool3 = nn.AvgPool2d(2, stride=2)


        self.flatten = Flatten()
        self.drop1 = nn.Dropout()
        self.line1 = nn.Linear(8192, 4096)
        self.relu7 = nn.ReLU(inplace=True)

        self.drop2 = nn.Dropout()
        self.line2 = nn.Linear(4096, 4096)
        self.relu8 = nn.ReLU(inplace=True)

        self.line3 = nn.Linear(4096, 20)
        self.fc = nn.Softmax()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool1(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.pool2(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.pool3(out)

        out = self.flatten(out)
        out = self.drop1(out)
        out = self.line1(out)
        out = self.relu7(out)

        out = self.drop2(out)
        out = self.line2(out)
        out = self.relu8(out)
        out = self.line3(out)
        return self.fc(out)

class MiniDecisionBlock(nn.Module):
    def __init__(self):
        super(MiniDecisionBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(256, 512, 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace = True)

        self.conv6 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool3 = nn.AvgPool2d(2, stride=2)


        self.flatten = Flatten()
        self.drop1 = nn.Dropout()
        self.line1 = nn.Linear(8192, 4096)
        self.relu7 = nn.ReLU(inplace=True)

        self.drop2 = nn.Dropout()
        self.line2 = nn.Linear(4096, 4096)
        self.relu8 = nn.ReLU(inplace=True)

        self.line3 = nn.Linear(4096, 100)
        self.fc = nn.Softmax()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool1(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.pool2(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.pool3(out)

        out = self.flatten(out)
        out = self.drop1(out)
        out = self.line1(out)
        out = self.relu7(out)

        out = self.drop2(out)
        out = self.line2(out)
        out = self.relu8(out)
        out = self.line3(out)
        return self.fc(out)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


### Check if GPU is available ####
torch.cuda.is_available()

#### Set Datatype ###
dtype = torch.FloatTensor # the CPU datatype
gpu_dtype = torch.cuda.FloatTensor

# Constant to control how frequently we print train loss
print_every = 100

def train(model, loss_fn, optimizer, num_epochs=1, loader_train=loader_train):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        model.cuda()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(gpu_dtype))
            #x_var = Variable(x.type(dtype))

            y_var = Variable(y.type(gpu_dtype).long())
            #y_var = Variable(y.type(dtype).long())

            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype), volatile=True)
        #x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return 100 * acc

def check_accuracy_decision(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype), volatile=True)
        #x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    if num_samples != 0:
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    else:
        print("Number of Samples = 0")
    return num_correct, num_samples

def split_data(model, loader, mode='train'):
    print("Splitting data into superclasses...")
    superclass_data = {}
    for i in range(20):
        superclass_data[i] = []
    superclass_y = {}
    for i in range(20):
        superclass_y[i] = []
    model.eval()
    if mode == 'train' or mode == 'val':
        #Is this the same for train and val?
        dataset = loader.dataset.train_data
        label_set = loader.dataset.train_labels
    if mode == 'test':
        dataset = loader.dataset.test_data
        label_set = loader.dataset.test_labels
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype), volatile=True)
        #x_var = Variable(x.type(dtype), volatile=True)

        y_var = Variable(y.type(gpu_dtype).long())
        #y_var = Variable(y.type(dtype).long())

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        for i in range(len(preds)):
            superclass_data[preds[i][0]].append(dataset[i])
            superclass_y[preds[i][0]].append(label_set[i])
    for key in (superclass_data.keys()):
        if superclass_data[key] != []:
            superclass_data[key] = np.stack(superclass_data[key])
        #superclass_y[key] = torch.stack(superclass_y[key])
    print("...")
    return superclass_data, superclass_y

# def augment_data(split_data, split_labels, num_augment = 3):
#     print ("Augmenting Data...")
#     cifar_compose = Compose([ToTensor(),
#                          TypeCast('float'),
#                          ChannelsFirst(),
#                          RangeNorm(0,1),
#                          RandomGamma(0.2,1.8),
#                          RandomBrightness(0.1, 0.3),
#                          RandomSaturation(0.5,0.9)])
#     for superclass in split_data.keys():
#         new_data = []
#         new_labels = []
#         data = split_data[superclass]
#         print (data.shape)
#         for i in range(num_augment):
#             print ("Augmentation Round ", i + 1)
#             augmented_data = cifar_compose(data)
#             print(augmented_data)
#             print(type(augmented_data))
#             new_data.append(augmented_data.numpy())
#             new_labels += split_labels
#         new_data = np.stack(new_data)
#         split_data[superclass] = new_data
#         split_labels[superclass] = new_labels
#     return split_data, split_labels


def create_loaders(split_tensors, split_labels, mode='train'):
    print("loading reclassified data...")
    training_set = []
    if mode == "train" or mode == "val":
        for superclass in split_tensors.keys():
            cifar_copy = copy.deepcopy(cifar100_train)
            cifar_copy.train_data = split_tensors[superclass]
            cifar_copy.train_labels = split_labels[superclass]
            loader = DataLoader(cifar_copy, batch_size=64, sampler=ChunkSampler(len(split_labels[superclass])))
            training_set.append(loader)
    elif mode == "val":
        for superclass in split_tensors.keys():
            cifar_copy = copy.deepcopy(cifar100_train)
            cifar_copy.train_data = split_tensors[superclass]
            cifar_copy.train_labels = split_labels[superclass]
            loader = DataLoader(cifar_copy, batch_size=64, sampler=ChunkSampler(len(split_labels[superclass])))
            training_set.append(loader)
    else:
        for superclass in split_tensors.keys():
            #new_num_train = math.ceil(len(split_labels[superclass]) * .9)
            #new_num_val = len(split_labels[superclass]) - new_num_train
            #change copy based on test or train
            cifar_copy = copy.deepcopy(cifar100_test)
            cifar_copy.test_data = split_tensors[superclass]
            cifar_copy.test_labels = split_labels[superclass]
            loader = DataLoader(cifar_copy, batch_size=64, sampler=ChunkSampler(len(split_labels[superclass])))
            training_set.append(loader)

    print("...")
    return training_set

def train_masses(training_set, val_set):
    model_list = []
    total_correct = 0
    total_samples = 0

    for superclass in range(len(training_set)):
        model = MiniDecisionBlock()
        loss_fn = nn.CrossEntropyLoss().type(dtype)
        optimizer = optim.SGD(model.parameters(), lr=.0075, momentum=.95)
        train(model, loss_fn, optimizer, loader_train=training_set[superclass], num_epochs=1)
        model_list.append(model)
        print("Checking Accuracy on SuperClass ", superclass, " :")
        if val_set[superclass] != []:
            correct, samples = check_accuracy_decision(model, val_set[superclass])
            total_correct += correct
            total_samples += samples
        else:
            print("No images were categorized as Superclass ", superclass, " in the validation set.")
    print("Total Accuracy: ", total_correct * 100 / float(total_samples), "%")
    return model_list


def run_test(model, model_list, loader_test):
    total_correct = 0
    total_samples = 0
    superclass_data, superclass_y = split_data(model, loader_test, mode='test')
    test_set = create_loaders(superclass_data, superclass_y, mode='test')
    for i in range (len(model_list)):
        correct, samples = check_accuracy_decision(model_list[i],test_set[i])
        total_correct += correct
        total_samples += samples
    print("Total Accuracy on Test Data: ", total_correct * 100. / total_samples, "%")
    return 1

### Define Model ###
start_time = time.time()
model = WideResNet(28, 20, widen_factor=10)

loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.SGD(model.parameters(), lr=.0075, momentum=.95)

##### Train Decision Network #####
train(model, loss_fn, optimizer, num_epochs=25)
check_accuracy(model, loader_val)
check_accuracy(model, loader_test)

##### Pass data through model #####
# Reload fine labels
cifar100_train.train_labels = train_fine_list
loader_train = DataLoader(cifar100_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))
cifar100_val.train_labels = train_fine_list
loader_val = DataLoader(cifar100_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
cifar100_test.test_labels = test_fine_list
loader_test = DataLoader(cifar100_test, batch_size=64)

#Split Data and Labels
super_data, super_y = split_data(model, loader_train, mode = 'train')
#super_data, super_y = augment_data(super_data, super_y)
val_data, val_y = split_data(model, loader_val, mode = 'val')

#Create Separate Classifiers
training_set = create_loaders(super_data, super_y, mode='train')
val_set = create_loaders(val_data, val_y, mode='val')
model_list = train_masses(training_set, val_set)


##Timing##
runtime = time.time() - start_time
print("Runtime: ", runtime)


#Test Model
run_test(model, model_list, loader_test)