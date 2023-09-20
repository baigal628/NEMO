import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from nanopore_dataset import create_sample_map
from nanopore_dataset import create_splits
from nanopore_dataset import load_csv
from nanopore_dataset import NanoporeDataset
from resnet1d import ResNet1D

from seqUtil import *

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

resnet1D = ResNet1D(
            in_channels=1,
            base_filters=128,
            kernel_size=3,
            stride=2,
            groups=1,
            n_block=8,
            n_classes=2,
            downsample_gap=2,
            increasefilter_gap=4,
            use_do=False)

def nntPredict(signals, device, model, weights_path, sigWindow = 400, method = 'mean'):
    '''
    Given a list of signals, return predicted modification scores.
    '''
    
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.to(device)
    
    # set input data
    input_tensor = torch.zeros((1, 1, 400)).to(device)
    sequence_tensor = torch.tensor(signals)
    
    sigRange = len(signals)
    probs = []
    
    for sigIdx in range(sigRange-sigWindow+1):
        input_tensor[:, :, :] = sequence_tensor[sigIdx:sigIdx+sigWindow]
        prob = model(input_tensor).sigmoid().item()
        probs.append(prob)
    return np.mean(probs)


def modelScores(refSeq, sigList, siglenList, sigStart,
                device, model, weights_path, outfile = '', 
                kmerWindow = 80, sigWindow = 400, modbase = ''):
    
    outFh = open(outfile, 'w')
    
    for pos in range(len(refSeq)):
        start = int(siglenList[sigStart+pos])
        end = int(siglenList[sigStart+pos+1]) + sigWindow
        
        signals = [float(s) for s in sigList[start:end]]
        
        seq = refSeq[pos:pos+kmerWindow]
        freq = baseCount(seq=seq, base = modbase)/len(seq)
        base_pos = basePos(seq, base = modbase)
        modBasePos = ','.join([str(b) for b in base_pos])
        
        prob = nntPredict(signals,device = device, model = model, weights_path = weights_path)
        if pos%500 == 0:
            print('prob:', prob)
            print('Predicitng modification at position: ', pos)
        out = '{seq}\t{prob}\t{freq}\t{base_pos}\n'.format(seq = seq, prob = prob, freq = freq, base_pos = modBasePos)
        outFh.write(out)
    print('Writing output to outfile')
    outFh.close()