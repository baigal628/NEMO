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
    
    probs = []
    
    #Here I omit +1 from len(signals)-sigWindow+1 because len(signals) already has one extra signal that corresponds to next kmer window.
    for sigIdx in range(len(signals)-sigWindow):
        input_tensor[:, :, :] = sequence_tensor[sigIdx:sigIdx+sigWindow]
        prob = model(input_tensor).sigmoid().item()
        probs.append(prob)
    
    return aggregate_scors(probs)

def tune_siganl(sigList, min_val=50, max_val=130):
    new_sigList = [max(min_val, min(max_val, float(signal))) for signal in sigList]
    return new_sigList

def assign_scores(strand, refSeq, modPositions, sigList, siglenList, sigLenList_init,
                  weights, model, device, tune = False, method = 'median', kmerWindow=80, signalWindow=400):
    
    # Position of As, relative to the reference
    modScores = {i:[] for i in modPositions}
    
    for pos in range(len(refSeq)):
        if pos % 500 ==0:
            print('Predicting at position:', pos)
        
        # 1. Fetch sequences with kmer window size, this step is optional
        # seq = refSeq[pos:pos+kmerWindow]
        # 2. Fetch signals with signal window size 
        pos_sigLenList_start = int(sigLenList_init)+pos
        pos_sigLenList_end = pos_sigLenList_start+1
        if pos_sigLenList_start<0: 
            start=0
        else:
            start = int(siglenList[pos_sigLenList_start])
        if len(sigList)-start< signalWindow:
            print('Reached the end of the signal.')
            break
        end = int(siglenList[pos_sigLenList_end])
        # If no signals aligned to this position. E.g. chrII 429016 is missed is eventalign output.
        if start == end:
            # print('No signal captured at position: ', pos)
            continue
        signals = [float(s) for s in sigList[start:end+signalWindow]]
        # 3. Get predicted probability score from machine learning model
        prob = nntPredict(signals,device = device, model = model, weights_path = weights)
        if len(signals) == signalWindow:
            print(start, end)
            break
        # 4. Assign predicted scores to each modPosition
        # modifiable positions [1,3,4,5,7,10,15,16,21,40]
        # kmer position is 2: [2:2+22]
        # modbase_left = 1
        # modbase_right = 9
        # modifiable position within kmer window [3,4,5,7,10,15,16,21]
        modbase_left = bisect.bisect_left(modPositions, pos)
        modbase_right = bisect.bisect_right(modPositions, pos+kmerWindow)
        modbase_count = modbase_right - modbase_left
        
        for p in range(modbase_left, modbase_right):
            modPosition = modPositions[p]
            # 4.1 Tune signals based on position of A and A content:
            if tune:
                strand = strand
                prob = model_scores(prob, modPosition, pos, modbase_count, strand)
            modScores[modPosition].append(prob)
    
    scores = []
    for mod in modScores:
        score = aggregate_scors(modScores[mod], method = method)
        scores .append(score)
    return scores