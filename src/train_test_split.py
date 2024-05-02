import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

# data set formatting
from nanopore_dataset import create_sample_map
from nanopore_dataset import create_splits
from nanopore_dataset import load_sigalign
from nanopore_dataset import load_parquet
from nanopore_dataset import NanoporeDataset

def add_parser(parser):
    parser.add_argument('--exp_id', default='test')
    parser.add_argument('--device', default='auto')
   
    # if you do not have a train validation split, specify the above
    parser.add_argument('--neg_data', type= str, default='', help='unmodified dechromatanized dna sequences, either a siglaign file or a parquet file format.')
    parser.add_argument('--pos_data', type= str, default='', help='fully modified dechromatinized dna sequences, either a siglaign file or a parquet file format.')
    parser.add_argument('--input_dtype', type= str, default='parquet', help='choose between sigalign or parquet. DEFAULT: parquet.')
    parser.add_argument('--train_split', type=float, default=0.6, help='fraction of data used for training model. DEFAULT: 0.6.')
    parser.add_argument('--val_split', type=float, default=0.2, help='fraction of data used for model validation. DEFAULT: 0.2.')
    parser.add_argument('--test_split', type=float, default=0.2, help='fraction of data used for testing model. DEFAULT: 0.2.')

    #input and output data preprocessing parameters
    parser.add_argument('--min_val', type=float, default=50) # Used to clip outliers
    parser.add_argument('--max_val', type=float, default=130) # Used to clip outliers
    parser.add_argument('--seq_len', type=int, default=400)
    parser.add_argument('--max_seqs', type=int, default=None)
    parser.add_argument('--outpath', type=str, default='./')


################################
# train, validation, test split #
#################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eventalign to sigalign file.')           
    add_parser(parser)
    args = parser.parse_args()
    
    ##############
    # set device #
    ##############
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print('Device type:', device)
    
    
    
    if args.input_dtype == 'sigalign':
        load_data = load_sigalign
    elif args.input_dtype == 'parquet':
        load_data = load_parquet

    print("Preparing unmodified...")
    print("Loading csv...")
    unmodified_sequences = load_data(args.neg_data,
                                        min_val=args.min_val,
                                        max_val=args.max_val,
                                        max_sequences=args.max_seqs)
    print("Creating sample map...")
    unmodified_sample_map = create_sample_map(unmodified_sequences,
                                            seq_len=args.seq_len)

    print("Creating splits...")
    unmodified_train, unmodified_val, unmodified_test = create_splits(unmodified_sequences,
                                                                    unmodified_sample_map,
                                                                    train_split=args.train_split,
                                                                    val_split=args.val_split,
                                                                    test_split=args.test_split,
                                                                    shuffle=True,
                                                                    seq_len=args.seq_len)
    print("Prepared.")

    print("Preparing modified...")
    print("Loading csv...")
    modified_sequences = load_data(args.pos_data,
                                    min_val=args.min_val,
                                    max_val=args.max_val,
                                    max_sequences=args.max_seqs)
    print("Creating sample map...")
    modified_sample_map = create_sample_map(modified_sequences,
                                            seq_len=args.seq_len)
    print("Creating splits...")
    modified_train, modified_val, modified_test = create_splits(modified_sequences,
                                                                modified_sample_map,
                                                                train_split=args.train_split,
                                                                val_split=args.val_split,
                                                                test_split=args.test_split,
                                                                shuffle=True,
                                                                seq_len=args.seq_len)
    print("Prepared.")

    ###############################
    # create torch data set class #
    ###############################

    print('Creating torch dataset class...')
    train_dataset = NanoporeDataset(unmodified_sequences,
                                    unmodified_train,
                                    modified_sequences,
                                    modified_train,
                                    device=device,
                                    synthetic=False,
                                    seq_len=args.seq_len)

    val_dataset = NanoporeDataset(unmodified_sequences,
                                unmodified_val,
                                modified_sequences,
                                modified_val,
                                device=device,
                                synthetic=False,
                                seq_len=args.seq_len)

    test_dataset = NanoporeDataset(unmodified_sequences,
                                unmodified_test,
                                modified_sequences,
                                modified_test,
                                device=device,
                                synthetic=False,
                                seq_len=args.seq_len)

    torch.save(val_dataset, f'{args.outpath}/val_dataset_{args.exp_id}.pt')
    torch.save(train_dataset, f'{args.outpath}/train_dataset_{args.exp_id}.pt')
    torch.save(test_dataset, f'{args.outpath}/test_dataset_{args.exp_id}.pt')
