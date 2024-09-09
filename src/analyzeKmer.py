import os as os
from scipy.signal import find_peaks
import pyarrow.parquet as pq
import pysam
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
plt.style.use('default')

def getKmerScores(bam, parquet, max_batch = ''):

    kmer_scores_for = {}
    kmer_scores_rev = {}

    print('reading bam file...')
    readToStrand = {}
    samfile = pysam.AlignmentFile(bam, "rb")
    for s in samfile:
        if s.is_mapped and not s.is_supplementary and not s.is_secondary:
            strand = 1
            if s.is_reverse:
                strand = -1
            readToStrand[s.query_name] = strand
    samfile.close()
    print('done.')

    print('reading parquet file...')
    parquet_file = pq.ParquetFile(parquet)
    print(f'{parquet_file.num_row_groups} total number of groups in current parquet file.')
    for z in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(z)
        print(f'total number of reads in this batch {batch.num_rows}')
        for i in tqdm(range(batch.num_rows)):
            readname =  batch['readcode'][i].as_py()
            if readname not in readToStrand: continue
            strand = readToStrand[readname]
            kmer = batch['qkmer'][i].as_py()
            signal_mean = np.mean([round(i,3) for i in batch['signal'][i].as_py()])
            
            if strand == 1:
                if kmer not in kmer_scores_for:
                    kmer_scores_for[kmer] = [signal_mean]
                else:
                    kmer_scores_for[kmer].append(signal_mean)
            else:
                if kmer not in kmer_scores_rev:
                    kmer_scores_rev[kmer] = [signal_mean]
                else:
                    kmer_scores_rev[kmer].append(signal_mean)
        if max_batch:
            if z == max_batch: break
    parquet_file.close()
    print('done.')
    return kmer_scores_for, kmer_scores_rev

def countKmerPeak(allkmers, pos_kmer_score_for, pos_kmer_score_rev, neg_kmer_score_for, neg_kmer_score_rev, outpath, prefix, prominence, distance):
    
    thislabel = ['positive control forward', 'positive control reverse', 'negative control forward', 'negative control reverse']
    colors = ['tab:purple', 'tab:pink', 'tab:blue', 'tab:green']
    outf = open(os.path.join(outpath, prefix+f'_signal_mean_hist.tsv'), 'w')
    outf.write('kmer\tpos_for\tpos_rev\tneg_for\tneg_rev\tnpeaks\n')
    sigpeakf = open(os.path.join(outpath, prefix+f'_kmer_with_multiple_peaks.tsv'), 'w')
    sigpeakf2 = open(os.path.join(outpath, prefix+f'_kmer_shifted_signals.tsv'), 'w')
    store = False
    for thiskmer in tqdm(allkmers):
        hists = []
        peaks = []
        thiskmer_scores = [pos_kmer_score_for[thiskmer], pos_kmer_score_rev[thiskmer], neg_kmer_score_for[thiskmer], neg_kmer_score_rev[thiskmer]]
        for i in range(len(thiskmer_scores)):
            hist, bin_edges = np.histogram(thiskmer_scores[i], range(0, 150, 1), density=True)
            if np.mean(hist[:30]) != 0: store=True
            peak, _ = find_peaks(hist, prominence=prominence, distance=distance, height = 0.005, width=3)
            hists.append(hist)
            peaks.append(len(peak))
        #     plt.bar(np.arange(0, 149, 1), hist, width=1, alpha=0.5, color=colors[i])
        #     plt.plot([np.arange(0, 149, 1)[x] for x in peak], [hist[x] for x in peak], 'x', label=f'{thislabel[i]} peaks', color=colors[i])
        # plt.xlim(0, 200)
        # plt.xlabel('signal picoampere (pA)')
        # plt.ylabel('density')
        # plt.title(thiskmer)
        # plt.legend()
        # outfig = os.path.join(outpath, prefix+f'_{thiskmer}_signal_mean_density.pdf')
        # plt.savefig(outfig, bbox_inches='tight')
        # plt.close()
        outf.write(thiskmer+'\t'+'\t'.join([','.join([str(x) for x in hist]) for hist in hists]) + '\t'+','.join([str(x) for x in peaks]) + '\n')
        if np.max(peaks) > 4:
            sigpeakf.write(thiskmer + '\n')
        if store:
            sigpeakf2.write(thiskmer + '\n')
    outf.close()
    sigpeakf.close()

def add_parser(parser):
    parser.add_argument('--posbam', type = str, default='', help = 'positive control bam file.')
    parser.add_argument('--negbam', type = str, default='', help = 'negative control bam file.')
    parser.add_argument('--posparq', type = str, default='', help = 'positive control parquet file.')
    parser.add_argument('--negparq', type = str, default='', help = 'negative control parquet file.')
    parser.add_argument('--outpath', type = str, default='./', help = 'output file path.')
    parser.add_argument('--prefix', type = str, default='', help = 'output file prefix')
    parser.add_argument('--max_batch', type = int, default=0, help = 'maximum number of batches to load from parqute')
    parser.add_argument('--min_prominence', type = float, default=0.001, help = 'minimum prominence for finding a peak.')
    parser.add_argument('--min_distance', type = int, default=10, help = 'minimum distance between two peaks.')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot modifications')           
    add_parser(parser)
    args = parser.parse_args()

    print('processing negative control data...')
    neg_kmer_score_for, neg_kmer_score_rev = getKmerScores(args.negbam, args.negparq, max_batch=args.max_batch)
    print('processing positive control data...')
    pos_kmer_score_for, pos_kmer_score_rev = getKmerScores(args.posbam, args.posparq, max_batch=args.max_batch)
    print('Start kmer level analysis...')
    allkmers = set(neg_kmer_score_for.keys()) & set(neg_kmer_score_rev.keys()) & set(pos_kmer_score_for.keys()) & set(pos_kmer_score_rev.keys())
    allkmers = sorted(allkmers)
    print(f'{len(allkmers)} number of shared kmers between positive and negative control data.')
    countKmerPeak(allkmers, pos_kmer_score_for, pos_kmer_score_rev, neg_kmer_score_for, neg_kmer_score_rev, args.outpath, args.prefix, args.min_prominence, args.min_distance)