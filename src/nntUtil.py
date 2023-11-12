import numpy as np
import torch
from nanoUtil import fetchSignal
import time



def aggregate_scors(scores, method):
    if method == 'mean':
        return np.nanmean(scores)
    elif method == 'median':
        return np.nanmedian(scores)
    elif method == 'min':
        return np.nanmin(scores)
    elif method == 'max':
        return np.nanmax(scores)
    
def nntPredict(signals, device, model, weights, signalWindow = 400, method = 'mean'):
    '''
    Given a list of signals, return predicted modification scores.
    '''
    
    model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    model.to(device)
    
    # set input data
    input_tensor = torch.zeros((1, 1, 400)).to(device)
    sequence_tensor = torch.tensor(signals)
    
    probs = []
    
    #Here I omit +1 from len(signals)-sigWindow+1 because len(signals) already has one extra signal that corresponds to next kmer window.
    for sigIdx in range(len(signals)-signalWindow):
        input_tensor[:, :, :] = sequence_tensor[sigIdx:sigIdx+signalWindow]
        prob = model(input_tensor).sigmoid().item()
        probs.append(prob)
    
    return aggregate_scors(probs, method = method)

def tune_siganl(sigList, min_val=50, max_val=130):
    new_sigList = [max(min_val, min(max_val, float(signal))) for signal in sigList]
    return new_sigList

def runNNT(readID, strand, bins, step, aStart, aEnd, sigList, sigLenList, kmerWindow, signalWindow, device, model, weight):
    
    print('Start processing read', readID)
    start_time = time.time()
    
    L = int(np.floor(aStart/step))
    R = int(np.floor(aEnd/step))
    binScores = {}
    
    for i in range(L, R+1):
        # 1. Fetch sequences with kmer window size, this step is optional
        # seq = refSeq[bin:bin+kmerWindow]
        
        # 2. Fetch signals with signal window size
        start = max(bins[i],aStart)
        end = min(start+kmerWindow, aEnd)
        
        print('Predicting at position:', start, '-',end)
        
        signals = fetchSignal(start-aStart, end-aStart, sigLenList, sigList, signalWindow)
        
        if signals == 'end':
            break
        # signals not enough (less than signalWindow) or not signals aligned to this region (should not happen theoraticaly)
        elif signals == 'del':
            continue
        else:
            # Get predicted probability score from machine learning model
            prob = nntPredict(signals, device = device, model = model, weights_path = weight, signalWindow = signalWindow)
            binScores[bins[i]] = prob

    total_time = "%.2f" % (time.time()-start_time)
    
    print('finished processing ', readID, ' in ', total_time, 's.')
    
    return binScores