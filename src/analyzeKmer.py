import os as os
from scipy.signal import find_peaks
import pyarrow.parquet as pq
import pysam
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
from scipy.stats import ks_2samp, gaussian_kde
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
plt.style.use('default')

def getKmerScores(parquet, bam='', max_batch='', kmerList='', bystrand=False, stat='mean', normalize=False, mean='', std = ''):

    if bystrand:
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
    
    else:
        kmer_scores = {}
    
    print('reading parquet file...')
    parquet_file = pq.ParquetFile(parquet)
    print(f'{parquet_file.num_row_groups} total number of groups in current parquet file.')
    for z in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(z)
        print(f'total number of reads in this batch {batch.num_rows}')
        for i in tqdm(range(batch.num_rows)):
            readname =  batch['readcode'][i].as_py()
            kmer = batch['qkmer'][i].as_py()
            if kmerList:
                if kmer not in kmerList: continue
            signals = batch['signal'][i].as_py()
            if normalize:
                signals = (np.array(signals)-mean)/std
            signal_mean = round(np.mean(signals), 3)
            signal_std = round(np.std(signals), 3)
            if stat == 'mean':
                signal_stat = signal_mean
            elif stat == 'std':
                signal_stat = signal_std
            if bystrand:
                if readname not in readToStrand: continue
                strand = readToStrand[readname]
                if strand == 1:
                    if kmer not in kmer_scores_for:
                        kmer_scores_for[kmer] = [signal_stat]
                    else:
                        kmer_scores_for[kmer].append(signal_stat)
                else:
                    if kmer not in kmer_scores_rev:
                        kmer_scores_rev[kmer] = [signal_stat]
                    else:
                        kmer_scores_rev[kmer].append(signal_stat)
            else:
                if kmer not in kmer_scores:
                    kmer_scores[kmer] = [signal_stat]
                else:
                    kmer_scores[kmer].append(signal_stat)
        if max_batch:
            if z == max_batch: break
    
    parquet_file.close()
    print('done.')
    if bystrand:
        return kmer_scores_for, kmer_scores_rev
    else:
        return kmer_scores


def comapreKmerDensity(kmer_scores, outpath, prefix, labels = ['positive control', 'negative control'], colors = ['tab:purple', 'tab:green'], plot_desnity=False, kmerlist = ''):

    print('calculating data range')
    lower_x = min(np.quantile(np.concatenate(list(kmer_scores[0].values())), 0.05), np.quantile(np.concatenate(list(kmer_scores[1].values())), 0.05))
    lower_x = np.floor(lower_x)
    upper_x = max(np.quantile(np.concatenate(list(kmer_scores[0].values())), 0.95), np.quantile(np.concatenate(list(kmer_scores[1].values())), 0.95))
    upper_x = np.ceil(upper_x)
    step = (upper_x-lower_x)/100
    print(f'range: ({lower_x}, {upper_x}), step: {step}')
    
    allkmers = set(kmer_scores[0].keys()) & set(kmer_scores[1].keys())
    allkmers = sorted(allkmers)
    print(f'{len(allkmers)} number of shared kmers between positive and negative control data.')
    
    outf = open(os.path.join(outpath, prefix+f'_signal_mean_density.tsv'), 'w')
    
    outf.write('kmer\tpos_density\tneg_density\tpos_count\tneg_count\tks_stat\tpval\n')

    sigpeakf = open(os.path.join(outpath, prefix+f'_kmer_with_diff_density.tsv'), 'w')
    nonsigpeakf = open(os.path.join(outpath, prefix+f'_kmer_without_diff_density.tsv'), 'w')
    colors = ['tab:purple', 'tab:green']
    labels = ['positive control', 'negative control']
    
    if kmerlist:
        allkmers = kmerlist
    for thiskmer in tqdm(allkmers):
        densities = []
        kmer_count = []
        # not enough kmer occurance to calculate density
        if len(kmer_scores[0][thiskmer]) <= 10 or len(kmer_scores[1][thiskmer]) <= 10:
            kmer_count = [len(kmer_scores[0][thiskmer]), len(kmer_scores[1][thiskmer])]
            outf.write(thiskmer+'\t.\t.\t' + '\t'.join(str(c) for c in kmer_count) + '\t.\t.\n')
            continue
        for i in range(2):
            kmer_count.append(len(kmer_scores[i][thiskmer]))
            try:
                kde = gaussian_kde(kmer_scores[i][thiskmer])
            except:
                print(thiskmer)
                continue
            x = np.arange(lower_x, upper_x, step)
            density = kde(x)
            densities.append(density)
            
        ks_stat, p_value = ks_2samp(kmer_scores[0][thiskmer], kmer_scores[1][thiskmer])
        if plot_desnity:
            for i in range(2):
                plt.plot(x, densities[i], label=f'{labels[i]} {kmer_count[i]}', color=colors[i])
                plt.bar(x, densities[i], color=colors[i], alpha = 0.6)
            plt.text(upper_x*0.8, 0, f'ks:{round(ks_stat, 3)}\npval:{round(p_value,3)}')
            plt.xlim(lower_x, upper_x)
            plt.xlabel('signal picoampere (pA)')
            plt.ylabel('density')
            plt.title(thiskmer)
            plt.legend()
            outfig = os.path.join(outpath, prefix+f'_{thiskmer}_signal_mean_density.pdf')
            plt.savefig(outfig, bbox_inches='tight')
            plt.show()
            plt.close()
        outf.write(thiskmer+'\t'+'\t'.join([','.join([str(round(x,3)) for x in density]) for density in densities]) + '\t'+ '\t'.join(str(c) for c in kmer_count) + '\t' + str(round(ks_stat,3)) + '\t' + str(round(p_value, 3)) + '\n')
        
        if p_value<0.001:
            sigpeakf.write(thiskmer + '\n')
        else:
            nonsigpeakf.write(thiskmer + '\n')
    outf.close()
    sigpeakf.close()
    nonsigpeakf.close()

def add_parser(parser):
    parser.add_argument('--posbam', type = str, default='', help = 'positive control bam file.')
    parser.add_argument('--negbam', type = str, default='', help = 'negative control bam file.')
    parser.add_argument('--posparq', type = str, default='', help = 'positive control parquet file.')
    parser.add_argument('--negparq', type = str, default='', help = 'negative control parquet file.')
    parser.add_argument('--outpath', type = str, default='./', help = 'output file path.')
    parser.add_argument('--prefix', type = str, default='', help = 'output file prefix')
    parser.add_argument('--stat', type = str, default='mean', help = 'using mean or standard deviation for summarizing signals.')
    parser.add_argument('--normalize', action='store_true', help = 'normalize signals using mean and standard deviation')
    parser.add_argument('--mean', type = float, default=0, help = 'sample mean used to normalize signals.')
    parser.add_argument('--std', type = float, default=0, help = 'sample std used to normalize signals.')
    parser.add_argument('--max_batch', type = int, default=0, help = 'maximum number of batches to load from parqute')
    # parser.add_argument('--min_prominence', type = float, default=0.001, help = 'minimum prominence for finding a peak.')
    # parser.add_argument('--min_distance', type = int, default=10, help = 'minimum distance between two peaks.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot modifications')           
    add_parser(parser)
    args = parser.parse_args()

    print('processing negative control data...')
    neg_kmer_scores = getKmerScores(args.negparq, args.negbam, max_batch=args.max_batch, stat=args.stat)
    print('processing positive control data...')
    pos_kmer_scores = getKmerScores(args.posparq, args.posbam, max_batch=args.max_batch, stat=args.stat)
    print('Start kmer level analysis...')
    
    comapreKmerDensity([pos_kmer_scores, neg_kmer_scores], args.outpath, args.prefix, stat=args.stat, normalize=args.normalize, mean=args.mean, std=args.std)