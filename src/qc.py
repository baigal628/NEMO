import os
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import numpy as np
import pysam
from bisect import bisect_left
from tqdm import tqdm
from scipy import stats

def getReadQual(bam):
    readToQual = {}
    samfile = pysam.AlignmentFile(bam, "rb")
    for s in samfile:
        if s.is_mapped and not s.is_supplementary and not s.is_secondary:
            readToQual[s.query_name] = s.query_qualities
    samfile.close()
    return readToQual

def getMeanReadQual(bam, predfile):
    all_qual, all_mod, all_readlen = [], [], []
    
    readToQual = getReadQual(bam)
    
    with open(predfile, 'r') as infile:
        for line in tqdm(infile):
            readname = line.strip().split('\t')[0]
            if readname not in readToQual: continue
            start = line.strip().split('\t')[3]
            scores = line.strip().split('\t')[-1]
            scores = [float(i) for i in scores.split(',')]
            avequal = np.mean(readToQual[readname])
            avemod = np.mean(scores)
            all_qual.append(avequal)
            all_mod.append(avemod)
            all_readlen.append(len(scores))
    return all_qual, all_mod, all_readlen

def plotReadQual(qual, mod, readlen, outpath, prefix, color = 'tab:blue'):
    
    readlen = [np.log10(r+1) for r in readlen]
    
    plt.plot(qual, mod, '.', color=color)
    plt.xlabel('average quality')
    plt.ylabel('average modifications')
    res = stats.spearmanr(qual, mod)
    plt.title(f'spearman cor:{round(res.statistic, 3)}\npvalue:{round(res.pvalue, 3)}')
    outfig = os.path.join(outpath, prefix+f'_qual_vs_mod.pdf')
    plt.savefig(outfig, bbox_inches='tight')
    plt.close()
    
    plt.plot(readlen, mod, '.', color=color)
    plt.xlabel('log10 (read length)')
    plt.ylabel('average modifications')
    res = stats.spearmanr(readlen, mod)
    plt.title(f'spearman cor:{round(res.statistic, 3)}\npvalue:{round(res.pvalue, 3)}')
    outfig = os.path.join(outpath, prefix+f'_readlength_vs_mod.pdf')
    plt.savefig(outfig, bbox_inches='tight')
    plt.close()
    
    plt.plot(readlen, qual, '.', color=color)
    plt.xlabel('log10 (read length)')
    plt.ylabel('average quality')
    res = stats.spearmanr(readlen, qual)
    plt.title(f'spearman cor:{round(res.statistic, 3)}\npvalue:{round(res.pvalue, 3)}')
    outfig = os.path.join(outpath, prefix+f'_readlength_vs_qual.pdf')
    plt.savefig(outfig, bbox_inches='tight')
    plt.close()
    
    plt.hist(mod, density = True, bins='auto', color=color)
    plt.xlabel('average mod score per read')
    plt.ylabel('density')
    outfig = os.path.join(outpath, prefix+f'_mod_density.pdf')
    plt.savefig(outfig, bbox_inches='tight')
    plt.close()
    
    plt.hist(qual, density = True, bins='auto', color=color)
    plt.xlabel('average quality score per read')
    plt.ylabel('density')
    outfig = os.path.join(outpath, prefix+f'_qual_density.pdf')
    plt.savefig(outfig, bbox_inches='tight')
    plt.close()
    
    plt.hist(readlen, density = True, bins=200, color=color)
    plt.xlabel('readlength score per read')
    plt.ylabel('density')
    outfig = os.path.join(outpath, prefix+f'_readlength_density.pdf')
    plt.savefig(outfig, bbox_inches='tight')
    plt.close()

def filterReadbyLen(predfile, outpath, prefix, cutoff = 1000):
    good_reads = open(os.path.join(outpath, prefix+f'_readlen_over_1k.tsv'), 'w')
    bad_reads = open(os.path.join(outpath, prefix+f'_readlen_under_1k.tsv'), 'w')
    with open(predfile, 'r') as infile:
        for line in tqdm(infile):
            readname = line.strip().split('\t')[0]
            scores = line.strip().split('\t')[-1]
            scores = [float(i) for i in scores.split(',')]
            if len(scores) >= cutoff:
                good_reads.write(line)
            else:
                bad_reads.write(line)
    good_reads.close()
    bad_reads.close()

def filterReadbyMod(predfile, outpath, prefix, cutoff = [20, 240]):
    good_reads = open(os.path.join(outpath, prefix+f'_mod_within_{cutoff[0]}_{cutoff[1]}.tsv'), 'w')
    bad_reads = open(os.path.join(outpath, prefix+f'_mod_outof_{cutoff[0]}_{cutoff[1]}.tsv'), 'w')
    with open(predfile, 'r') as infile:
        for line in tqdm(infile):
            readname = line.strip().split('\t')[0]
            scores = line.strip().split('\t')[-1]
            scores = [float(i) for i in scores.split(',')]
            if np.mean(scores) <= cutoff[0] or np.mean(scores) >= cutoff[1]:
                bad_reads.write(line)
            else:
                good_reads.write(line)
    good_reads.close()
    bad_reads.close()