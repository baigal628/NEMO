from bamUtil import getAlignedReads
from nanoUtil import parseEventAlign, parseSigAlign
from nntUtil import runNNT
from resnet1d import ResNet1D
from plotUtil import plotAllTrack
from trackUtil import predToBedGraph
import numpy as np
import argparse
import torch
import multiprocessing


parser = argparse.ArgumentParser('NEMO')
subparsers = parser.add_subparsers(help='sub-command help')
parser_train = subparsers.add_parser('train', help='train neural network model on your data.')
parser_train.add_argument('--pos', '-p', type=str)
parser_train.add_argument('--neg', '-n', type=str)

parser_predict = subparsers.add_parser('predict', help='train neural network model on your data.')
# class input
parser_predict.add_argument('--region', '-r', default = 'all', type=str, action='store', help='genomic coordinates to perform modification predictions. E.g. chrI:2000-5000 or chrI or all (for whole genome).')
parser_predict.add_argument('--bam', '-b', default = '', type=str, action='store', help='path to sorted, indexed, and binarized alignment file.')
parser_predict.add_argument('--genome', '-g', default = '', type=str, action='store', help='reference genome fasta file')
parser_predict.add_argument('--eventalign', '-e', default='', type=str, action='store', help='nanopolish eventalign file.')
parser_predict.add_argument('--sigalign', '-s', default='', type=str, action='store', help='sigalign file if sigAlign file already exist. If not, must provide eventalign to generate sigAlign file.')
parser_predict.add_argument('--readlist', '-rl', default='', type=str, action='store', help='readId list created along with sigalign file.')

# class output
parser.add_argument('--outpath', '-o', default='./', type=str, action='store', help='path to store the output files.')
parser.add_argument('--prefix', '-p', default='nemo', type=str, action='store', help='prefix of output file names.')

# modPredict input
parser.add_argument('--model', '-m', default='resnet1d', type=str, action='store', help='deep neural network meodel used for prediction. DEFAULT: resnet1d.')
parser.add_argument('--weight', '-w', default='', type=str, action='store', help='path to the trained model.')
parser.add_argument('--threads', '-t', default=4, type=int, action='store', help='number of threads.')
parser.add_argument('--step', '-step', default=20, type=int, action='store', help='step to bin the region.')
parser.add_argument('--kmerWindow', '-kw', default=75, type=int, action='store', help='kmer window size to extend bin.')
parser.add_argument('--signalWindow', '-sw', default=400, type=int, action='store', help='signal Window size to feed into the model.')
parser.add_argument('--load', '-l', default=50, type=int, action='store', help='number of reads to load into each iterations. Each iteration will output a file.')
parser.add_argument('--threshold', '-threshold', default=(0.3, 0.55, 0.6), action='store', help='prediction value above this threshold willl be called as modified (1).')

#  plot input
parser.add_argument('--prediction', '-pred', default = '', type=str, action='store', help='path to prediction file from modification prediction results.')
parser.add_argument('--ncluster', '-nc', default = 3, type=int, action='store', help='number of kmean clusters.')
parser.add_argument('--gtf', '-gtf', default = '', type=str, action='store', help='path to General Transfer Format (GTF) file.')
parser.add_argument('--refbdg', '-rbdg', default = '', type=str, action='store', help='path to ground truth ot short read bedgraph.')
parser.add_argument('--predbdg', '-pbdg', default = '', type=str, action='store', help='path to aggregated prediction bedgraph from predToBedGraph call.')
parser.add_argument('--pregion', '-pregion', default = '', type=str, action='store', help='region to plot. Can be gene name of the pre defined gene regions.')

args = parser.parse_args()

class findNemo:

    '''
    class findNemo: predict small molecule modifications from nanopore long-read sequencing data.
    '''
    
    def __init__(self, region, bam, genome, outpath, prefix, eventalign, sigalign, readlist, step):
        
        '''
        self:
            self.alignment: dict, stores reference an reads mapped to region.
            self.chrom: str, chromosome name.
            self.qStart: int, query start position.
            self.qEnd: int, query end position
            self.outpath: str, path to store the output files.
        Input:
            region: genomic coordinates to perform modification predictions. Format: 'chrI:2000-5000' or 'chrI'.
            bam: sorted, indexed, and binarized alignment file.
            genome: reference genome fasta file.
        Output:
            outpath: path to store the output files.
            prefix: prefix of output file names.
            eventalign: nanopolish eventalign file.
            sigalign: sigalign file if sigAlign file already exist. If not, must provide eventalign to generate sigAlign file.
        '''

        # Fetch reads aligned to the region
        print('Collecting reads mapped to ', region, ' ...')
        self.region = region
        self.alignment, self.chrom, self.qStart, self.qEnd = getAlignedReads(bam, region, genome)
        self.outpath = outpath
        self.prefix = prefix
        self.step = step

        if isinstance(self.chrom, list):
            self.bins = {}
            for i in range(len(self.chrom)):
                self.bins[self.chrom[i]] = np.arange(self.qStart[i], self.qEnd[i], self.step)
        else:    
            self.bins = np.arange(self.qStart, self.qEnd, self.step)
        
        # Index reads to avoid storing the long readnames.
        if readlist:
            print('readling read list...')
            myreadlist = {}
            with open(readlist, 'r') as rl:
                for line in rl:
                    line = line.strip().split('\t')
                    myreadlist[line[0]] = line[1]
            
            self.reads = {r:myreadlist[r] for r in self.alignment}
        else:
            self.reads = {r:i for r,i in zip(self.alignment, range(len(self.alignment)))}
            readFh = open(outpath + prefix + '_' + region + '_readID.tsv', 'w')
            for k,v in self.reads.items(): readFh.write('{read}\t{index}\n'.format(read = k, index = v))
            readFh.close()
        
        self.alignment = {int(self.reads[r]):self.alignment[r] for r in self.reads}
        # print(self.alignment)
        # Store the readname index match into a file.

        print(len(self.reads), " reads mapped to ", region)
        
        self.gene_regions = {
            'PHO5': 'chrII:429000-435000',
            'CLN2': 'chrXVI:66400-67550',
            'HMR': 'chrIII:290000-299000',
            'AUA1': 'chrVI:114000-116000',
            'EMW1': 'chrXIV:45000-50000',
            'NRG2': 'chrII:370000-379000',
            'RDN37': 'chrXII:450300-459300'
            }
    
    def eventTosigalign(self, eventalign):
        
        assert eventalign != '', "eventalign file not provided."
        
        self.sigalign = self.outpath + self.prefix + '_' + str(self.region) + '_sig.tsv'
        parseEventAlign(eventAlign = eventalign, reads = self.reads, outfile = self.sigalign)
    
    def doWork(self, work):
        (readID, strand, bins, step, aStart, aEnd, qStart, sigList, sigLenList, kmerWindow, signalWindow, device, model, weight) = work
        
        scores = runNNT(readID, strand, bins, step, aStart, aEnd, qStart, sigList, sigLenList, kmerWindow, signalWindow, device, model, weight)
        
        return scores
    
    def modPredict(self, model, weight, threads, kmerWindow, signalWindow, load):
        
        print('Start predicting modified positions...')
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        torch.multiprocessing.set_start_method('spawn')
        # only one model available
        models = {
            'resnet1d': ResNet1D(
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
            }
        
        print('Device type: ', device)
        
        # Total work to be done are stored in a list
        works = [(readID, strand, self.bins, self.step, aStart, aEnd, self.qStart, sigList, sigLenList, kmerWindow, signalWindow, device, models[model], weight) 
                  for readID, aStart, aEnd, strand, sigList, sigLenList in parseSigAlign(self.sigalign, self.alignment)]
        # Use the specified threads number or maximum available CPU cores
        num_processes = min(threads, multiprocessing.cpu_count())
        
        for x in range(0, len(works), load):
            # split total work by loads 
            works_per_load = works[x:x+load]
            # open the pool for multiprocessing
            pool = multiprocessing.Pool(processes=num_processes)
            # Use the pool.map() function to process reads in parallel
            outs = pool.map(self.doWork, works_per_load)
            
            # Close the pool to release resources
            pool.close()
            pool.join()
            
            # Write the results from current work load
            predOut = self.outpath + self.prefix + '_'  + str(self.region) + '_' + str(x) + '_prediction.tsv'
            predOutFh = open(predOut, 'w')
            for r in range(len(outs)):
                out = outs[r]
                if out:
                    readID = works_per_load[r][0]
                    strand = works_per_load[r][1]
                    bin_start = next(iter(out))
                        
                    predOutFh.write('{readID}\t{strand}\t{bin_start}\t{scores}\n'.format(readID = readID, strand = strand, bin_start = bin_start, scores = ','.join(map(str, out.values()))))
            predOutFh.close()
            print('Prediction scores were writted in ',predOut, '.')


    def exportBedgraph(self, prediction, threshold):
                
        predToBedGraph(prediction, self.chrom, self.bins, self.step, threshold[1], self.outpath, self.prefix)
        print('Done exporitng bedgraph.')
    
    def plotTrack(self, prediction, gtf, pregion, threshold, ncluster):
        if pregion in self.gene_regions:
            myregion = self.gene_regions[pregion]
        else:
            myregion = pregion
        
        print('plotting region:', myregion)
        
        plotAllTrack(prediction=prediction, gtfFile=gtf, bins = self.bins, step = self.step, 
                              outpath=self.outpath, prefix=self.prefix, pregion = myregion, 
                              colorRange =threshold, ncluster=ncluster)
        
        print('Done plotting genome track.')

if __name__ == '__main__':
    myprediction = findNemo(args.region, args.bam, args.genome, args.outpath, args.prefix, args.eventalign, args.sigalign, args.readlist, args.step)
    
    assert args.mode in ['init', 'train', 'predict', 'plot']
    
    if args.mode == 'train':
        print('Done Preprocessing!')
    
    # predict modification sites from aligned signals
    elif args.mode == 'predict':
        if not args.prediction:
            myprediction.modPredict(args.model, args.weight, args.threads, args.kmerWindow, args.signalWindow, args.load)
    
    # make bedgraph from prediction
    elif args.mode == 'callbdg':
        print('Writing prediction to bedgraph...')
        myprediction.exportBedgraph(args.prediction, args.threshold)
    
    elif args.mode == 'plot':
        myprediction.plotTrack(args.prediction, args.gtf, args.pregion, args.threshold, args.ncluster)