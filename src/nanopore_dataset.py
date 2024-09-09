"""
Data-loading utils and dataset class for nanopore data
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pyarrow.parquet as pq
import random
from bisect import bisect_left, bisect_right

def tune_signal(sigList, min_val=-50, max_val=150):
    new_sigList = [max(min_val, min(max_val, round(float(signal), 3))) for signal in sigList]
    return new_sigList

# Load all data into sequences
def load_csv(filename, min_val=-50, max_val=150, max_sequences=None):
    '''
    load csv signal file, with 'START' as idicator of new read.
    '''
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

def load_sigalign(filename, min_val=-50, max_val=150, max_sequences=None):
    '''
    read siglaign file and reformat into a seq of signals.
    '''
    sequences = []
    with open(filename, 'r') as sigFile:
        header = sigFile.readlines(1)
        for line in tqdm(sigFile):
            line = line.strip().split('\t')
            signals = tune_signal(sigList = line[4].split(','), min_val=min_val, max_val=max_val)
            sequences.append(signals)
            if max_sequences is not None:
                if len(sequences) == max_sequences:
                    break
    return sequences

def load_parquet(filename, min_val=-50, max_val=150, max_sequences=None):
    '''
    read pyarrow parquet file and reformat into a seq of signals.
    input:
        filename: path to parquet file.
        min_val, max_val: threshold to tune signals
        max_sequences: maximum number of reads to load per batches
    '''
    sequences = []
    parquet_file = pq.ParquetFile(filename)
    print(f'{parquet_file.num_row_groups} total number of groups in current parquet file.')
    for z in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(z)
        print(z, 'group')
        if max_sequences:
            # randomly select max_sequences reads from current batches
            max_seq = min(batch.num_rows, max_sequences)
            myranges = random.sample(range(batch.num_rows), max_seq)
        else:
            myranges = range(batch.num_rows)
        for i in myranges:
            signals = batch['signals'][i].as_py()
            # siglenperkmer = batch['siglenperkmer'][i].as_py()
            signals = tune_signal(signals, min_val=min_val, max_val=max_val)
            sequences.append(signals)
            # print('len:', len(signals))
    return sequences

# Compute map for generating samples on-the-fly
def create_sample_map(sequences, seq_len, step):
    sample_map = []
    for i, sequence in enumerate(tqdm(sequences)):
        for j in range(len(sequence) - (seq_len - 1)):
            if step:
                if j%step==0:
                    sample_map.append([i, j])
    return sample_map


# Create splits
def create_splits(sequences,
                  sample_map,
                  train_split,
                  val_split,
                  test_split,
                  shuffle,
                  seq_len,
                  step):

    # Compute n_samples per split
    n_samples = len(sample_map)
    n_train_samples = np.ceil(train_split * n_samples)
    n_val_samples = np.ceil(val_split * n_samples)
    print(f'total nsamples: {n_samples}, ntrain: {n_train_samples}, nval: {n_val_samples}')

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

        sample_idx += np.floor((len(sequences[sequence_idx]) - seq_len)/step)+1
        print(sample_idx)

    print(f'total reads: {len(sequence_idxs)}, ntrain: {len(train_sequence_idxs)}, nval: {len(val_sequence_idxs)}, ntest: {len(test_sequence_idxs)}')

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
                 device,
                 synthetic,
                 seq_len):

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

def create_pred_sample_map(sigalign, seq_len, readlist, step):
    '''
    create_pred_sample_map function reads signal alignment file and prepare sample map for training nnt model.
    input:
        sigalign: signal alignment file
        seq_len: input signal dimension for nnt
        readlist: a list of read indeces to make prediction on
        step: step size for storing each sample
    output:
        sample_map: a list of sample map sets. each set consists of (read index,  signal index, chromsome, alignment start)
        sequences: a list of sequencing reads.
    '''
    sequences = {}
    sample_map = {}
    with open(sigalign) as infile:
        header = infile.readlines(1)
        for line in infile:
            line = line.strip().split('\t')
            readIdx = line[0]
            strand = int(line[1])
            chrom = line[2]
            sequence = tune_signal(line[4].split(','))
            if readlist:
                if readIdx not in readlist:
                    continue
            if chrom not in sample_map:
                sample_map[chrom] = []
            sequences[readIdx] = sequence
            kmer_move_table = list(map(int, line[5].split(',')))
            end_5 = int(line[3])
            siganl_move = 0
            kmer_move = 0
            for siganl_move in range(len(sequence)-seq_len+1):
                pos_end = bisect_left(kmer_move_table, siganl_move+seq_len-1)
                (left, right) = (end_5+kmer_move, end_5+pos_end) if strand == 1 else (end_5-pos_end, end_5-kmer_move)
                # skip signals for a faster run
                if step:
                    if siganl_move%step == 0:
                        # print(readIdx, strand, siganl_move, chrom, left, right)
                        sample_map[chrom].append((readIdx, strand, siganl_move, chrom, left, right))
                # move one signal at a time
                else:
                    sample_map[chrom].append((readIdx, siganl_move, chrom, left, right))           
                while siganl_move == kmer_move_table[kmer_move]:
                    kmer_move +=1
    return (sample_map, sequences)

def create_pred_sample_map_parquet(parquet, myreads, seq_len, step, max_sequences = ''):
    '''
    create_pred_sample_map function reads signal alignment file and prepare sample map for training nnt model.
    input:
        sigalign: signal alignment file
        seq_len: input signal dimension for nnt
        readlist: a list of read indeces to make prediction on
        step: step size for storing each sample
    output:
        sample_map: a list of sample map sets. each set consists of (read index,  signal index, chromsome, alignment start)
        sequences: a list of sequencing reads.
    '''
    
    parquet_file = pq.ParquetFile(parquet)
    sequences = {}
    sample_map = {}
    for z in tqdm(range(parquet_file.num_row_groups)):
        batch = parquet_file.read_row_group(z)
        print(f'{z} group has {batch.num_rows} reads')
        if max_sequences:
            max_seq = min(batch.num_rows, max_sequences)
            myranges = random.sample(range(batch.num_rows), max_seq)
        else:
            myranges = range(batch.num_rows)
        for i in myranges:
            readIdx = batch['readname'][i].as_py()
            if readIdx not in myreads:
                continue
            chrom = batch['chr'][i].as_py()
            # this is end_3 in reverse read
            end_5 = batch['startpos'][i].as_py()
            kmer_move_table =  list(map(int, batch['siglenperkmer'][i].as_py()))
            sequence = tune_signal(batch['signals'][i].as_py())
            sequences[readIdx] = sequence
            strand = myreads[readIdx][-1]
            if strand == -1:
                end_5 = end_5 + len(kmer_move_table)-1
            if chrom not in sample_map:
                sample_map[chrom] = []
            
            kmer_move = 0

            # align signals to reference position
            for signal_move in range(len(sequence)-seq_len+1):
                # pos_end: signal 3' end position on reference genome relative to the 5' alignstart
                # kmer_move: signal 5' end position on reference genome relative to the 5' alignstart
                pos_end = bisect_right(kmer_move_table, signal_move+seq_len-1)
                (left, right) = (end_5+kmer_move, end_5+pos_end) if strand == 1 else (end_5-pos_end, end_5-kmer_move)
                # skip signals for a faster run
                if step:
                    if signal_move%step == 0:
                        sample_map[chrom].append((readIdx, strand, signal_move, chrom, left, right))
                # move one signal at a time
                else:
                    sample_map[chrom].append((readIdx, strand, signal_move, chrom, left, right))           
                while signal_move == kmer_move_table[kmer_move]:
                    kmer_move +=1
    return (sample_map, sequences)

def downsample_sample_map(sample_map, n):
    '''
    downsample n number of dataset from sample_map
    '''
    dsample_map = {}
    for chrom in sample_map:
        chrom_map = [sample_map[chrom][i] for i in random.sample(range(len(sample_map[chrom])), n)]
        dsample_map[chrom] = chrom_map
    return dsample_map