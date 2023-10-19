"""
Data-loading utils and dataset class for nanopore data
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# Load all data into sequences
def load_csv(filename,
             min_val=50,
             max_val=130,
             max_sequences=None):
    sequences = []
    sequence = []
    with open(filename, 'r') as f:
        for line in tqdm(f):
            data = line.strip()
            if data == 'START':
                if len(sequence) > 0:
                    sequences.append(sequence)
                sequence = []
            elif data != '':
                #data will be refomatted from 50 to 130
                val = max(min_val, min(max_val, float(data)))
                sequence.append(val)
                if max_sequences is not None:
                    if len(sequences) == max_sequences:
                        break
    return sequences


# Compute map for generating samples on-the-fly
def create_sample_map(sequences, seq_len=400):
    sample_map = []
    for i, sequence in enumerate(tqdm(sequences)):
        for j in range(len(sequence) - (seq_len - 1)):
            sample_map.append([i, j])
    return sample_map


# Create splits
def create_splits(sequences,
                  sample_map,
                  train_split=0.8,
                  val_split=0.2,
                  test_split=0.0,
                  shuffle=False,
                  seq_len=400):

    # Compute n_samples per split
    n_samples = len(sample_map)
    n_train_samples = np.ceil(train_split * n_samples)
    n_val_samples = np.ceil(val_split * n_samples)

    # Collect sequence indices
    sequence_idxs = np.arange(len(sequences))
    if shuffle:
        np.random.shuffle(sequence_idxs)
    sample_idx = 0
    train_sequence_idxs = set()
    val_sequence_idxs = set()
    test_sequence_idxs = set()
    for sequence_idx in tqdm(sequence_idxs):
        if sample_idx <= n_train_samples:
            train_sequence_idxs.add(sequence_idx)
        elif sample_idx <= n_train_samples + n_val_samples:
            val_sequence_idxs.add(sequence_idx)
        else:
            test_sequence_idxs.add(sequence_idx)

        sample_idx += len(sequences[sequence_idx]) - (seq_len - 1)

    # Create sample map splits
    train_sample_map = []
    val_sample_map = []
    test_sample_map = []
    for sample in tqdm(sample_map):
        if sample[0] in train_sequence_idxs:
            train_sample_map.append(sample)
        elif sample[0] in val_sequence_idxs:
            val_sample_map.append(sample)
        elif sample[0] in test_sequence_idxs:
            test_sample_map.append(sample)
        else:
            raise

    return train_sample_map, val_sample_map, test_sample_map



# Dataset class
class NanoporeDataset(Dataset):

    def __init__(self,
                 unmodified_sequences,
                 unmodified_sample_map,
                 modified_sequences,
                 modified_sample_map,
                 device='cpu',
                 synthetic=False,
                 seq_len=400):

        self.unmodified_sequences = unmodified_sequences
        self.unmodified_sample_map = unmodified_sample_map
        self.modified_sequences = modified_sequences
        self.modified_sample_map = modified_sample_map
        self.device = device
        self.synthetic = synthetic
        self.seq_len = seq_len

    def __len__(self):
        return len(self.unmodified_sample_map) + len(self.modified_sample_map)

    def __getitem__(self, idx):

        if idx < len(self.unmodified_sample_map):
            if self.synthetic:
                sample = torch.ones(self.seq_len, device=self.device).unsqueeze(0)
            else:
                i, j = self.unmodified_sample_map[idx]
                sample = torch.tensor([self.unmodified_sequences[i][j:j+self.seq_len]],
                                      device=self.device)
            label = torch.tensor([0.], device=self.device)
        else:
            if self.synthetic:
                sample = torch.zeros(self.seq_len, device=self.device).unsqueeze(0)
            else:
                modified_idx = idx - len(self.unmodified_sample_map)
                i, j = self.modified_sample_map[modified_idx]
                sample = torch.tensor([self.modified_sequences[i][j:j+self.seq_len]],
                                      device=self.device)
            label = torch.tensor([1.], device=self.device)

        return sample, label

    def get_seq_idx(self, idx):

        if self.synthetic:
            return None

        if idx < len(self.unmodified_sample_map):
            seq_idx, sample_idx = self.unmodified_sample_map[idx]
        else:
            modified_idx = idx - len(self.unmodified_sample_map)
            seq_idx, sample_idx = self.modified_sample_map[modified_idx]

        return seq_idx
