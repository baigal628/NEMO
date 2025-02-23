import sys
from tqdm import tqdm
import random
import time
import argparse
import torch
import os
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
from nanopore_dataset import tune_signal, create_pred_sample_map, create_pred_sample_map_parquet
import multiprocessing
from functools import partial
import numpy as np
from bamUtil import getAlignedReads, idxToReads
import tempfile
import subprocess
import shutil
import matplotlib.pyplot as plt


from resnet1d import ResNet1D
from nanopore_convnet import NanoporeConvNet
from nanopore_transformer import NanoporeTransformer

def aggregate_scores(scores, method, thred = 0.5):
    '''
    different methods to summarize a list of scores.
    '''
    if method == 'mean':
        return np.nanmean(scores)
    elif method == 'median':
        return np.nanmedian(scores)
    elif method == 'min':
        return np.nanmin(scores)
    elif method == 'max':
        return np.nanmax(scores)
    elif method == 'bin':
        return np.sum([1 if i >= thred else 0 for i in scores ])/len(scores)

class NanoporeDataset(Dataset):

    def __init__(self,
                 pred_sample_map,
                 sequences,
                 device,
                 seq_len):

        self.pred_sample_map = pred_sample_map
        self.sequences = sequences
        self.seq_len = seq_len
        self.device = device

    def __len__(self):
        return len(self.pred_sample_map)

    def __getitem__(self, idx):

        readIdx, strand, signalIdx, chrom, start, end = self.pred_sample_map[idx]
        sample = torch.tensor([self.sequences[readIdx][signalIdx:signalIdx+self.seq_len]], device=self.device)
        
        return sample, readIdx, strand, chrom, start, end

def skewed_t_distribution(x, loc = 11 , scale = 10, df = 5, min_dist = 0.7):

    # Calculate the probability score y based on the t-distribution PDF
    y = t.pdf((x - loc) / scale, df) / t.pdf(0, df)

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('Probability score (y)')
    plt.title('Left-skewed t-distribution')
    plt.grid(True)
    plt.show()

    return {k:max(v, min_dist) for k,v in zip(x,y)}


def predictMod(dataloader, model_type, device, seq_len, weight, kmer_len, max_batches, mean, std, time_batch = False):

    pred_out = {}
    pred_out_list = []
    if model_type == 'convnet':
        model = NanoporeConvNet(input_size=seq_len).to(device)
    elif model_type == 'resnet':
        model = ResNet1D(
                    in_channels=1,
                    base_filters=128,
                    kernel_size=3,
                    stride=2,
                    groups=1,
                    n_block=8,
                    n_classes=2,
                    mean=mean, 
                    std=std,
                    downsample_gap=2,
                    increasefilter_gap=4,
                    use_do=False).to(device)
    elif model_type == 'transformer':
        model = NanoporeTransformer(d_model=128,
                                    dim_feedforward=256,
                                    n_layers=6,
                                    n_head=8).to(device)
    elif model_type == 'phys':
        model = ResNet1D(
                    in_channels=1,
                    base_filters=128,
                    kernel_size=16,
                    stride=2,
                    groups=32,
                    n_block=48,
                    n_classes=2,
                    mean=mean, 
                    std=std,
                    downsample_gap=6,
                    increasefilter_gap=12,
                    use_do=False).to(device)

    model.load_state_dict(torch.load(weight, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    # summary(model, (1, seq_len), device = device)
    print("Created model and moved to the device.")
    with torch.no_grad():
        idx = 0
        # sample, readIdx, strand, chrom, start, end
        for sample, readIdx, strand, chrom, start, end in tqdm(dataloader):
            if max_batches:
                if idx > max_batches:
                    break
            start_time = time.time()
            sample.to(device)
            pred = model(sample).sigmoid()
            pred_time = time.time()-start_time
            for i in range(len(pred)):
                thispred = int(pred[i].item()*256)
                thisstart = start[i].item()
                thisend = end[i].item()
                thisreadIdx = readIdx[i]
                thisstrand = int(strand[i].item())
                pred_out_list.append(thispred)
                # no need to add kmerlen because thisend is end of signal chunk
                for thispos in range(thisstart, thisend):
                    # print(thisstart, thisend+kmer_len-1, thispos, thispred)
                    if chrom[i] not in pred_out: pred_out[chrom[i]] = {(thisreadIdx, thisstrand):{thispos:[thispred]}}
                    elif (thisreadIdx, thisstrand) not in pred_out[chrom[i]]: pred_out[chrom[i]][(thisreadIdx, thisstrand)] = {thispos:[thispred]}
                    elif thispos not in pred_out[chrom[i]][(thisreadIdx, thisstrand)]: pred_out[chrom[i]][(thisreadIdx, thisstrand)][thispos] = [thispred]
                    else: pred_out[chrom[i]][(thisreadIdx, thisstrand)][thispos].append(thispred)
            if time_batch:
                if idx%100 == 0:
                    print(f'predicted {idx} batch in: {pred_time} s!')
                    print(f'stored {idx} batch in: {time.time()-pred_time} s!')
            idx +=1
    return pred_out, pred_out_list

def predToBed12(predout, outfile, cutoff, save_raw=True):
    rgb = '0,0,0'
    outf = open(outfile, 'w')
    if save_raw:
        outfr = open(outfile.split('.bed')[0]+'.tsv', 'w')
    intcutoff = int(cutoff*256)
    print(f'using {cutoff} as cutoff to binarize value and call nucleosomes...')
    for chrom, read_strands in predout.items():
        for read_strand in read_strands:
            read = read_strand[0]
            strand = '+' if read_strand[1] == 1 else '-'
            sortedread = sorted(read_strands[read_strand].items())
            thickStart, thickEnd = sortedread[0][0], sortedread[-1][0]
            thisblockStart=0
            # add start of the reads
            blockStart=[0]
            blockSizes=[1]
            blockCount=1
            poss = []
            scoress = []
            for (pos, scores) in sortedread:
                score = np.mean(scores)
                if save_raw:
                    if poss:
                        poss.append(pos-prepos)
                        prepos=pos
                    else:
                        poss.append(0)
                        prepos = pos
                    scoress.append(score)
                if pos ==  thickStart:
                    continue
                # closed chromatin (occupied)
                if score < intcutoff:
                    # start of a block
                    if not thisblockStart:
                        thisblockStart = pos-thickStart+1
                        blockStart.append(thisblockStart-1)
                        blockCount +=1
                        thisblockSizes = 1
                    else:
                        thisblockSizes+=1
                # open chromatin (modified)
                else:
                    if thisblockStart:
                        blockSizes.append(thisblockSizes)
                        thisblockStart=0
            
            # block continues to the end of the read
            if thisblockStart:
                blockSizes.append(thisblockSizes)
                thisblockStart=0
                blockCount+=1
            
            # add end of the reads
            blockStart.append(thickEnd-thickStart)
            blockSizes.append(1)
            blockCount+=1
            
            blockStart = ','.join(str(i) for i in blockStart)
            blockSizes = ','.join(str(i) for i in blockSizes)
            bed_out = f'{chrom}\t{thickStart}\t{thickEnd}\t{read}\t{round(score*1000)}\t{strand}\t{thickStart}\t{thickEnd}\t{rgb}\t{blockCount}\t{blockSizes}\t{blockStart}\n'
            outf.write(bed_out)
            if save_raw:
                allpos = ','.join(str(i) for i in poss)
                allscores = ','.join(str(i) for i in scoress)
                raw_out = f'{read}\t{chrom}\t{strand}\t{thickStart}\t{thickEnd}\t{allpos}\t{allscores}\n'
                outfr.write(raw_out)
    outf.close()
    outfr.close()

def pred(chrom, sample_map, sequences, model_type, batch_size, seq_len, device, weight, max_batches, 
         cutoff, tmpdir, prefix, kmer_len, mean=80, std=16, return_pred=False, make_bed=True):
    '''
    make prediction for one chromsome.
    '''
    tstart = time.time()
    print(f'Loading {chrom} data into pytorch...')
    pred_dataset = NanoporeDataset(sample_map[chrom], sequences, device, seq_len)
    pred_dataloader = DataLoader(pred_dataset, batch_size = batch_size, shuffle=False)
    predout, pred_out_list = predictMod(pred_dataloader, model_type, device, seq_len, weight, kmer_len, max_batches, mean, std, cutoff)
    if make_bed:
        bed12_out = os.path.join(tmpdir, prefix + '_' + chrom + '.bed')
        predToBed12(predout, bed12_out, cutoff)
        print(f'Prediction saved for {chrom}\n in {round(time.time()-tstart, 3)}s')
    if return_pred:
        return predout, pred_out_list
    else:
        del pred_dataset, pred_out_list, pred_dataloader, predout

def add_parser(parser):
    parser.add_argument('--parquet', type = str, default='', help = 'signal alignment file in parquet format (R10 data).')
    parser.add_argument('--sigalign', type = str, default='', help = 'signal alignment file in sigalign format (R9 data).')
    parser.add_argument('--bam', type = str, default='', help = 'BAM alignment file')
    parser.add_argument('--region', type = str, default='all', help = 'region to call prediction.')
    parser.add_argument('--readID', type = str, default='', help = 'read to idx tsv file.')
    parser.add_argument('--ref', type = str, default='', help = 'reference genome.')
    parser.add_argument('--cutoff', type = float, default=0.5, help = 'cutoff value to separate pos and neg prediction.')
    parser.add_argument('--seq_len', type = int, default = 400, help = 'input signal length. DEFUALT:400.')
    parser.add_argument('--kmer_len', type = int, default = 9, help = 'kmer length in the pore. Ususally it is 6 for ONT R9 and 9 for ONT R10. DEFUALT:9.')
    parser.add_argument('--step', type = int, default=10, help = 'step size to take for creating sample map. DEFUALT:0.')
    parser.add_argument('--readlist', nargs="*", default=[], help = 'a list of readIdx to make predictions')
    parser.add_argument('--device', type = str, default='auto', help = 'device type for pytorch. DEFUALT: auto.')
    parser.add_argument('--weight', type = str, default='', help = 'path to the model weights.')
    parser.add_argument('--thread', type = int, default=1, help = 'number of thread to use. DEFAULT:1.')
    parser.add_argument('--outpath', type = str, default='', help = 'output path.')
    parser.add_argument('--prefix', type = str, default='', help = 'outfile prefix.')
    parser.add_argument('--batch_size', type = int, default='', help = '')
    parser.add_argument('--model_type', type = str, default='resnet', help = '')
    parser.add_argument('--sample_mean', type = float, default=0, help = 'sample mean to normalize the data')
    parser.add_argument('--sample_std', type = float, default=0, help = 'sample standard deviation to normalize the data')
    parser.add_argument('--max_batches', type = int, default=0, help = 'maximum batches to process per chromsome.')
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='predit modification ')           
    add_parser(parser)
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print('Device type:', device)
    
    torch.multiprocessing.set_start_method('spawn', force=True)
    print('Creating sample map...')
    tstart = time.time()
    
    if args.sigalign:
        myreads = ''
        if args.region:
            assert args.bam, args.ref
            print(f'Reading bam file and getting reads aligned to {args.region}')
            myreads, chrom, start, end = idxToReads(args.bam, args.region, args.ref, args.readID)
            readlist = [i for i in myreads]
            print(readlist[:10])
            print(f'Collected total number of reads: {len(readlist)}')
        sample_map, sequences = create_pred_sample_map(args.sigalign, 
                                                    seq_len=args.seq_len, 
                                                    readlist=args.readlist, 
                                                    step=args.step)
    
    if args.parquet:
        print(f'predicting on region: {args.region}...')
        assert args.bam, args.ref
        myreads, chrom, start, end = getAlignedReads(args.bam, args.region, args.ref)

        sample_map, sequences = create_pred_sample_map_parquet(args.parquet, myreads, args.seq_len, args.step)
    
    print(f'Sample map created in {round(time.time()-tstart, 3)}s!')

    if not args.sample_mean:
        print('calculatint sample mean and std...')
        mymean = round(np.mean([item for sublist in sequences.values() for item in sublist]), 0)
        mystd = round(np.std([item for sublist in sequences.values() for item in sublist]),0)
        print(f'sample mean: {mymean}, sample std: {mystd}')
    
    else:
        mymean = args.sample_mean
        mystd = args.sample_std
        print(f'using user provided mean {mymean} and std {mystd}...')
    
    tmp_dir = tempfile.mkdtemp(dir=args.outpath)
    n_cores = min(args.thread, multiprocessing.cpu_count())  # Adjust number of cores if necessary
    print(f'Number of cores: {n_cores}')
    try:
        pool = multiprocessing.Pool(processes=n_cores)
        # chrom, sample_map, sequences, model_type, batch_size, seq_len, device, weight, max_batches, 
        #      cutoff, tmpdir, prefix, kmer_len, return_pred=False, make_bed=True
        pool.map(partial(pred,
                        sample_map=sample_map,
                        sequences=sequences,
                        model_type = args.model_type,
                        batch_size=args.batch_size, 
                        seq_len=args.seq_len, 
                        device=device, 
                        weight=args.weight, 
                        max_batches=args.max_batches,
                        cutoff=args.cutoff,
                        tmpdir=tmp_dir, 
                        prefix=args.prefix,
                        kmer_len=args.kmer_len,
                        mean=mymean, std=mystd,
                        return_pred=False), list(sample_map.keys()))
        pool.close()
        pool.join()
    except Exception as e:
        print(f"An error occurred: {e}")
        shutil.rmtree(tmp_dir)
        sys.exit(1) 
    print('aggregating bed files...')
    output_file = os.path.join(args.outpath, args.prefix+'.bed')
    bed_files = os.path.join(tmp_dir, args.prefix+'_chr*.bed')
    subprocess.run(f'cat {bed_files} > {output_file}', shell=True)

    print('aggregating tsv files...')
    output_file = os.path.join(args.outpath, args.prefix+'.tsv')
    bed_files = os.path.join(tmp_dir, args.prefix+'_chr*.tsv')
    subprocess.run(f'cat {bed_files} > {output_file}', shell=True)

    shutil.rmtree(tmp_dir)
    print('Done aggregating files.')
    print('Finished!')