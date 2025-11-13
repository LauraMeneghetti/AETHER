'''Module implementing the dataset for handling respiratory data.'''

import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class RespiratoryDataset(Dataset):
    '''
    Class implementing the custom dataset for handling respiratory data.
    '''
    def __init__(self, intervals, labels, augment=False):
        '''
        :param np.array intervals: intervalsof the respiratory events
        :param np.array labels: true labels of the intervals
        :param bool augment: If True, augmentation is enabled (during training).
            otherwise is not applied.
        '''
        self.X = intervals
        self.y = labels
        self.seq_lengths = [len(seq) for seq in self.X]
        self.augment = augment # Enable augmentation only during training


    def __len__(self):
        '''Function giving the length of the sequences in the dataset
        :return: number of sequences in the dataset
        :rtype: int'''
        return len(self.X)

    def augment_signal(self, signal):
        """Applies random augmentation (noise + shifting)
        :param np.array signal: signal to be augmented
        :return: augmented signal
        :rtype: np.array
        """
        if np.random.rand() < 0.5:
            signal = signal + np.random.normal(0, 0.01, size=signal.shape)  # Add noise
        if np.random.rand() < 0.5:
            shift = np.random.randint(-5, 5)
            signal = np.roll(signal, shift)  # Shift the signal
        return signal
    
    def __getitem__(self, idx):
        ''' Function giving the idx elements of the dataset.
        :param int idx: selected index
        :return: selected intervals, related labels and lenghts of the sequences
        :rtype: np.array, np.array, np.array
        '''

        # Apply augmentation only during training
        sequence = self.X[idx]
        if self.augment:
            sequence = self.augment_signal(sequence)
            sequence = torch.tensor(sequence, dtype=torch.float32)
        return sequence, self.y[idx], self.seq_lengths[idx]

    

