import numpy as np
import bisect
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

def aggregate_scors(scores, method='mean'):
    if method == 'mean':
        return np.nanmean(scores)
    elif method == 'median':
        return np.nanmedian(scores)
    elif method == 'min':
        return np.nanmin(scores)
    elif method == 'max':
        return np.nanmax(scores)
    
def nntPredict(signals, device, model, weights_path, sigWindow = 400):
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
    
    return aggregate_scors(probs)

def tune_siganl(sigList, min_val=50, max_val=130):
    new_sigList = [max(min_val, min(max_val, float(signal))) for signal in sigList]
    return new_sigList

def assign_scores(readID, sigList, siglenList, sigStart, modbase, alignemnt, 
                  weights, model, device, 
                  tune = False, method = 'median', kmerWindow=80, signalWindow=400):
    
    refSeq = alignemnt['ref']
    # Position of As, relative to the reference
    modPositions = basePos(refSeq, base = modbase)
    modScores = {i:[] for i in modPositions}
    
    for pos in range(len(refSeq)-signalWindow+1):
        if pos % 500 ==0:
            print('Predicting at position:', pos)
        
        # 1. Fetch sequences with kmer window size, this step can be skipped later
        seq = refSeq[pos:pos+kmerWindow]
        
        # 2. Fetch signals with signal window size 
        pStart_sigLenList = sigStart+pos-1
        if pStart_sigLenList<0: 
            start=0
        else:
            start = int(siglenList[pStart_sigLenList])
        end = int(siglenList[sigStart+pos])-1+signalWindow
        signals = [float(s) for s in sigList[start:end]]

        # 3. Get predicted probability score from machine learning model
        prob = nntPredict(signals,device = device, model = model, weights_path = weights)
        
        # 4. Assign predicted scores to each modPosition
        # modifiable positions [1,3,4,5,7,10,15,16,21,40]
        # kmer position is 2: [2:2+22]
        # modbase_left = 0
        # modbase_right = 9
        # modifiable position within kmer window [3,4,5,7,10,15,16,21]
        modbase_left = bisect.bisect_left(modPositions, pos)
        modbase_right = bisect.bisect_right(modPositions, pos+kmerWindow)
        modbase_count = modbase_right - modbase_left
        
        for p in range(modbase_left, modbase_right):
            modPosition = modPositions[p]
            # 4.1 Tune signals based on position of A and A content:
            if tune:
                strand = alignemnt[readID][1]
                prob = model_scores(prob, modPosition, pos, modbase_count, strand)
            modScores[modPosition].append(prob)
    
    for mod in modScores:
        modScores[mod] = aggregate_scors(modScores[mod], method = method)
    return modScores