
import time
import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

import memory_function

####### Layers ##################


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
        self.fc3   = nn.Linear(84, 100)

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

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


#Root location for datasets
root = "./datasets"
omniglot_root = "./datasets/omniglot"


######## CIFAR-100 Test #############

NUM_TRAIN = 49000
NUM_VAL = 1000

cifar100_train = dset.CIFAR100(root, train=True, download=True,
                           transform=T.ToTensor())
loader_train = DataLoader(cifar100_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))

cifar100_val = dset.CIFAR100(root, train=True, download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(cifar100_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

cifar100_test = dset.CIFAR100(root, train=False, download=True,
                          transform=T.ToTensor())

loader_test = DataLoader(cifar100_test, batch_size=64)


##### OMNIGLOT #########

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

print("Unpacking Omniglot...")
train_dict = unpickle(omniglot_root + "/train_omni.pkl")
test_dict = unpickle(omniglot_root + "/test_omni.pkl")

train_images = torch.from_numpy(train_dict[b"images"])
train_labels = torch.from_numpy(train_dict[b"labels"].astype("float"))
print(train_images.size())
print(train_labels.size())

test_images = torch.from_numpy(test_dict[b"images"])
test_labels = torch.from_numpy(test_dict[b"labels"].astype("float"))
print(test_images.size())
print(test_labels.size())

NUM_OMNI_TRAIN = 70000
NUM_OMNI_VAL = 7120

omni_train_dset = TensorDataset(train_images, train_labels)
omni_loader_train = DataLoader(omni_train_dset, batch_size=64, sampler=ChunkSampler(NUM_OMNI_TRAIN, 0))

omni_test_dset = TensorDataset(test_images, test_labels)
omni_loader_test = DataLoader(omni_test_dset, batch_size=64, sampler=ChunkSampler(NUM_OMNI_VAL, NUM_OMNI_TRAIN))

### Check if GPU is available ####
torch.cuda.is_available()

#### Set Datatype ###
dtype = torch.FloatTensor # the CPU datatype
gpu_dtype = torch.cuda.FloatTensor

# Constant to control how frequently we print train loss
print_every = 100

def train(model, loss_fn, optimizer, num_epochs=1, memory = False):
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
            if memory:
                prediction = memory.forward(scores)

            loss = loss_fn(scores, y_var)
            if memory:
                memory_loss = memory_function.memory_loss(memory, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))
                if memory:
                    print('Memory loss: ', memory_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if memory:
                memory_function.memory_update(memory, prediction, y_var)


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
        # x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)
        scores = memory.forward(scores)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return 100 * acc


##### Initialize Model ######
start_time = time.time()

model = LeNet()
memory = memory_function.Memory(49000, 100)
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.SGD(model.parameters(), lr=.0075, momentum=.95)


##### Train Network #####
train(model, loss_fn, optimizer, num_epochs=10, memory = memory)
check_accuracy(model, loader_val)

runtime = time.time() - start_time
print("Runtime: ", runtime)