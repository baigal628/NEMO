from tqdm import tqdm
import time
import argparse
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
from nntUtil import tune_signal

from resnet1d import ResNet1D
from nanopore_convnet import NanoporeConvNet
from nanopore_transformer import NanoporeTransformer

def add_parser(parser):
    parser.add_argument('--readlist', type = str, default='', help = 'eventalign file.')
    parser.add_argument('--outpath', type = str, default='./', help = 'output path.')
    parser.add_argument('--prefix', type = str, default='', help = 'outfile prefix.')
    parser.add_argument('--bed', type = str, default='', help = '')
    parser.add_argument('--region', type = str, default='', help = '')


def IdxToReads(bam, region, ref, readID):
    print('readling read list...')
    readsToIdx = {}
    with open(readID, 'r') as infile:
        for line in infile:
            line = line.strip().split('\t')
            # readname: line[0] idx: line[1]
            readsToIdx[line[0]] = line[1]
    alignment, chrom, qStart, qEnd = getAlignedReads(bam, region, ref)
    myreads = {r:reads[r] for r in alignment}

def create_pred_sample_map(sigalign, seq_len, readlist = '', step = 0):
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
    sample_map = []
    idx = 0
    with open(sigalign) as infile:
        for line in infile:
            line = line.strip().split('\t')
            sequence = tune_signal(line[3].split(','))
            chrom = line[1]
            readIdx = line[0]
            if readlist:
                if readIdx not in readlist:
                    continue
            sequences[readIdx] = sequence
            kmer_move_table = line[4].split(',')
            start = int(line[2])
            siganl_move = 0
            kmer_move = 0
            for i in range(len(sequence)-seq_len+1):
                siganl_move +=1
                # skip signals for a faster run
                if step:
                    if siganl_move%step == 0:
                        # add (readIdx,  signalIdx, chrom, start)
                        sample_map.append((readIdx, i, chrom, start+kmer_move))
                # move one signal at a time
                else:
                    sample_map.append((readIdx, i, chrom, start+kmer_move))
                while siganl_move == int(kmer_move_table[kmer_move]):
                    kmer_move +=1
            idx +=1
    return (sample_map, sequences)

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

        readIdx, signalIdx, chrom, start = self.pred_sample_map[idx]
        sample = torch.tensor([self.sequences[readIdx][signalIdx:signalIdx+self.seq_len]],
                                  device=self.device)
        
        return sample, readIdx, chrom, start
    
def predictMod(dataloader, model_type, device, seq_len, weight, window, max_batches = ''):

    pred_out = {}
    
    if model_type == 'convnet':
        model = NanoporeConvNet(input_size=args.seq_len).to(device)
    elif model_type == 'resnet':
        model = ResNet1D(
                    in_channels=1,
                    base_filters=128,
                    kernel_size=3,
                    stride=2,
                    groups=1,
                    n_block=8,
                    n_classes=2,
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
                    downsample_gap=6,
                    increasefilter_gap=12,
                    use_do=False).to(device)

    model.load_state_dict(torch.load(weight, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    # summary(model, (1, seq_len), device = device)
    print("Created model and moved to the device.")
    with torch.no_grad():
        idx = 1
        for sample, readIdx, chrom, start in tqdm(dataloader):
            if max_batches:
                if idx > max_batches:
                    break
            start_time = time.time()
            sample.to(device)
            pred = model(sample).sigmoid()
            pred_time = time.time()-start_time
            for i in range(len(pred)):
                thispred = round(pred[i].item())
                thisstart = start[i].item()
                for i in range(len(window)):
                    thispos = thisstart+i
                if chrom[i] not in pred_out: pred_out[chrom[i]] = {readIdx[i]:{thispos:[thispred]}}
                elif readIdx[i] not in pred_out[chrom[i]]: pred_out[chrom[i]][readIdx[i]] = {thispos:[thispred]}
                elif thispos not in pred_out[chrom[i]][readIdx[i]]: pred_out[chrom[i]][readIdx[i]][thispos] = [thispred]
                else: pred_out[chrom[i]][readIdx[i]][thispos].append(thispred)
            if idx%10 == 0:
                print(f'predicted {idx} batch in: {pred_time} s!')
                print(f'stored {idx} batch in: {time.time()-pred_time} s!')
            idx +=1
    return pred_out

def predToBed12(predout, skip, outfile):
    strand = '.'
    rgb = '0,0,0'
    outf = open(outfile, 'w')
    for chrom, reads in predout.items():
        for read in reads:
            sortedread = sorted(reads[read].items())
            thickStart = sortedread[0][0]
            thickEnd = sortedread[-1][0]
            blockStart = [i[0]-thickStart for i in sortedread if round(np.mean(i[1])) == 1]
            blockCount = len(blockStart)
            blockSizes = np.zeros(blockCount, dtype = int) + skip
            if not blockStart:
                blockStart, blockCount, blockSizes = [0],1,[1]
            blockStart = ','.join(str(i) for i in blockStart)
            blockSizes = ','.join(str(i) for i in blockSizes)
            bed_out = f'{chrom}\t{thickStart}\t{thickEnd}\t{read}\t0\t{strand}\t{thickStart}\t{thickEnd}\t{rgb}\t{blockCount}\t{blockSizes}\t{blockStart}\n'
            outf.write(bed_out)
    outf.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='eventalign to sigalign file.')           
    add_parser(parser)
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print('Device type:', device)
    
    sample_map, sequences = create_pred_sample_map(args.sigalign, args.seq_len, precise = False)
    pred_dataset = NanoporeDataset(sample_map, sequences, device, args.seq_len)
    pred_dataloader = DataLoader(pred_dataset, batch_size=256, shuffle=True)
    predout = predictMod(pred_dataloader, 'resnet', device, args.seq_len, args.weight, max_seq = args.max_seq)
    
    predToBed12(predout, 75, '/private/home/gabai/Add-seq/data/train/pos_test_data.bed')