import os
import random
import _pickle as pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.sampler as sampler
from torchvision import transforms, utils

DATA_FILE_FORMAT = os.path.join(os.getcwd(), '%s_omni.pkl')

class OmniglotDataset(Dataset):
    """Omniglot dataset."""

    def __init__(self, data_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(DATA_FILE_FORMAT % data_file, "rb") as f:
            processed_data = pickle.load(f)
        self.images = np.vstack([np.expand_dims(np.expand_dims(image, axis=0), axis=0) for image in processed_data['images']])
        self.images = self.images.astype('float32')
        self.images /= 255.0

        self.labels = processed_data['labels']
        self.labels = self.labels.astype('int64')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx, :, :, :]
        label = self.labels[idx]
        sample = [torch.from_numpy(image), label]
        return sample

def random_index(seed, N):
    """ Args: seed - initial index, N - maximum index
        Return: A random index between [0, N] except for seed
    """
    offset = random.randint(1, N-1)
    idx = (seed + offset) % N 
    assert(seed != idx)
    return idx

class SiameseDataset(Dataset):
    """Siamese Dataset dataset."""

    def __init__(self, filepath):
        """
        Args:
            filepath (string): path to data file
            Data format - list of characters, list of images, (row, col, ch) numpy array normalized between (0.0, 1.0)
            Omniglot dataset - Each language contains a set of characters; Each character is defined by 20 different images
        """
        with open(filepath, "rb") as f:
            processed_data = pickle.load(f)

        self.data = dict()
        for image, label in zip(processed_data['images'], processed_data['labels']):
            if label not in self.data:
                self.data[label] = list()
            img = np.expand_dims(image, axis=0).astype('float32')
            img /= 255.0
            self.data[label].append(img)
        self.num_categories = len(self.data)
        self.category_size = len(self.data[processed_data['labels'][0]])

    def __len__(self):
        return self.num_categories

    def __getitem__(self, idx):
    	raise NotImplementedError


