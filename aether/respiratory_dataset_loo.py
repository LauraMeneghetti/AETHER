'''Module implementing the dataset for handling respiratory data for the LOO case.'''
import torch
from torch.utils.data import Dataset, DataLoader
import random

class RespiratoryDatasetLOO(Dataset):
    '''
    Class implementing the custom dataset for handling respiratory data in the LOO case.
    '''    
    def __init__(self, intervals, labels, patient_ids):
        '''
        :param np.array intervals: intervalsof the respiratory events
        :param np.array labels: true labels of the intervals
        :param np.array patient_ids: ids of the patients connected to the intervals
        '''

        self.X = intervals
        self.y = labels
        self.seq_lengths = [len(seq) for seq in self.X]
        self.patients_ids = patient_ids

    def __len__(self):
        '''Function giving the length of the sequences in the dataset
        :return: number of sequences in the dataset
        :rtype: int'''
        return len(self.X)
    
    def __getitem__(self, idx):
        ''' Function giving the idx elements of the dataset.
        :param int idx: selected index
        :return: selected intervals, related labels and lenghts of the sequences
        :rtype: np.array, np.array, np.array, np.array
        '''
        return self.X[idx], self.y[idx], self.seq_lengths[idx], self.patients_ids[idx]