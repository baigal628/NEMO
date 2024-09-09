from tqdm import tqdm
import sys
from seqUtil import fetchSize
from plot import map_score_to_window, plotModTrack
from scipy import stats
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd


def gettssWithGene(bed, genome, window, col = {'chrom':0, 'start':1, 'end':2, 'gene':3, 'strand':5}):
    
    tssposs = {}
    hw = window/2
    genomeSize = fetchSize(genome)
    with open(bed, 'r') as infile:
        for line in infile:
            if 'track' not in line:
                line = line.strip().split('\t')
                if len(line) == 1:
                    continue
                chr, dir = line[col['chrom']], line[col['strand']]
                if chr not in genomeSize:
                    continue
                pos = int(line[col['start']]) if dir == '+' else int(line[col['end']])
                gene = line[col['gene']]
                if chr not in tssposs:
                    tssposs[chr] = []
                left=int(max(0, pos-hw))
                right=int(min(pos+hw, genomeSize[chr]))
                
                tssposs[chr].append((left, pos, right, dir, gene))
        return tssposs
    

def gettssAgg(agg, in_window, out_window):
    agg_window = []
    inhw = int(in_window/2)
    outhw = int(out_window/2)
    with open(agg, 'r') as infile:
        for line in infile:
            line = line.strip().split('\t')
            if int(line[0]) >= inhw-outhw and int(line[0])<= inhw+outhw:
                agg_window.append(float(line[1]))
    return agg_window


def computeGeneAgg(predfile, tssposs, window, readnames = '', cutoff = '', strand_specific=False, hmm_model = ''):
    
    hw = int(window/2)
    cov_gene = {chrom:{gene:0 for gene in tssposs[chrom]} for chrom in tssposs}
    agg_scores_gene = {chrom:{gene:[[] for i in range(window+1)] for gene in tssposs[chrom]} for chrom in tssposs}
    
    with open(predfile, 'r') as infile:
        for line in tqdm(infile):
            chrom = line.strip().split('\t')[1]
            if chrom not in tssposs:
                continue
            readname = line.strip().split('\t')[0]
            if readnames:
                if readname not in readnames: continue
            strand = 1 if line.strip().split('\t')[2] == '+' else -1
            start = int(line.strip().split('\t')[3])
            scores = line.strip().split('\t')[-1]
            scores = [float(i) for i in scores.split(',')]
            sortedread = [(start+i,j) for i, j in enumerate(scores)]
            result = list(map(lambda x: map_score_to_window(x, strand=strand, sortedread=sortedread, hw=hw, cutoff=cutoff, strand_specific=strand_specific, odd=False, hmm_model=hmm_model), tssposs[chrom]))
            for r in range(len(result)):
                if result[r] != -1:
                    gene = tssposs[chrom][r]
                    for i, j in zip(result[r][0], result[r][1]):
                        agg_scores_gene[chrom][gene][i].append(j)
                    # enough coverage for this read promoter region
                    if len(result[r][0]) > int(window*2/3):
                        cov_gene[chrom][gene]+=1
    return cov_gene, agg_scores_gene


def scoreGenebyTss(agg_gene, cov_gene, tssAgg, outpath, prefix):
    outfile = open(outpath+f'{prefix}_gene_tss_score_coverage_window{int(len(tssAgg))-1}.tsv', 'w')
    outfile.write('geneid\tspearman_cor\tpearson_cor\tpval\tcoverage\tchr\ttss\tstrand\n')
    for chrom in agg_gene:
        for gene in tqdm(agg_gene[chrom]):
            agg_scores = [np.mean(x) if len(x) > 0 else 0 for x in agg_gene[chrom][gene]]
            if np.sum(agg_scores) == 0: continue
            res_spear = stats.spearmanr(tssAgg, agg_scores)
            res_pearson = stats.pearsonr(tssAgg, agg_scores)
            spear_cor = res_spear.statistic
            pear_cor = res_pearson.statistic
            pval = round(res_pearson.pvalue, 3)
            cov = cov_gene[chrom][gene]
            outfile.write(f'{gene[4]}\t{spear_cor}\t{pear_cor}\t{pval}\t{cov}\t{chrom}\t{gene[1]}\t{gene[3]}\n')
    outfile.close()

def plotGeneLoci(predfile, genetsspos, window, outpath, prefix, gtfFile, ncluster, min_cor = 0.8, min_cov = 15, cragne=[20, 120]):

    generank = pd.read_table(genetsspos)
    generank = generank.sort_values(by='pearson_cor', ascending=False)
    
    
    plt.hist(generank['spearman_cor'], bins = 20, density=True)
    plt.xlabel('sphe chrom gene spearman correlation with well positioned +1 nuc')
    plt.ylabel('density')
    plt.savefig(outpath+f'{prefix}_gene_tss_spearman.pdf')
    plt.close()

    plt.hist(generank['pearson_cor'], bins = 20, density=True)
    plt.xlabel('sphe chrom gene pearson correlation with well positioned +1 nuc')
    plt.ylabel('density')
    plt.savefig(outpath+f'{prefix}_gene_tss_spearman.pdf')
    plt.close()
    
    plt.hist(generank['coverage'], bins = 20, density=True)
    plt.xlabel('sphe chrom gene coverage')
    plt.ylabel('density')
    plt.savefig(outpath+f'{prefix}_gene_tss_coverage.pdf')
    plt.close()

    generank_wellpos = generank[(generank['pearson_cor'] >= min_cor) & (generank['coverage'] >= min_cov)]
    
    hw = int(window/2)
    for i in range(generank_wellpos.shape[0]):
        geneid= generank_wellpos.iloc[i]['geneid']
        print(f'gene: {geneid}...')
        chrom = generank_wellpos.iloc[i]['chr']
        tss = int(generank_wellpos.iloc[i]['tss'])
        pregion = f'{chrom}:{tss-hw}-{tss+hw}'
        plotModTrack(predfile, pregion, ncluster=ncluster, outpath=outpath, prefix= f'{prefix}_{geneid}', gtfFile=gtfFile, cutoff = '', xticks_space = 150, na_thred=0.3, crange=cragne)

def add_parser(parser):
    parser.add_argument('--predfile', type = str, default='', help = 'prediction output file.')
    parser.add_argument('--nuc_agg', type = str, default='', help = 'aggregated scores at +1 nuc')
    parser.add_argument('--bed', type = str, default='', help = 'read to idx tsv file.')
    parser.add_argument('--ref', type = str, default='', help = 'reference genome.')
    parser.add_argument('--window', type = int, default=600, help = 'window size around +1 nuc to compute spearman correlation.')
    parser.add_argument('--inwindow', type = int, default=2000, help = 'input +1 nuc aggregation file window size')
    parser.add_argument('--crange', nargs='+', type = int, default=[20,120], help = 'color range for plotting modification tracks.')
    parser.add_argument('--outpath', type = str, default='', help = 'output path.')
    parser.add_argument('--prefix', type = str, default='', help = 'outfile prefix.')
    parser.add_argument('--genetsspos', type = str, default='', help = 'gene tss nuc positioning score and coverage file. ')
    parser.add_argument('--gtfFile', type = str, default='', help = 'gene annotation file')
    parser.add_argument('--ncluster', type = int, default=1, help = 'gene annotation file')
    parser.add_argument('--min_cor', type = float, default=0.8, help = 'minimum pearson correlation to keep for genes to plot.')
    parser.add_argument('--min_cov', type = int, default=15, help = 'minimum coverage to keep for genes to plot.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot modifications')
    add_parser(parser)
    args = parser.parse_args()
    
    if not args.genetsspos:
    
        tssAgg = gettssAgg(args.nuc_agg, args.inwindow, args.window)

        tssposs = gettssWithGene(args.bed, args.ref, args.window)

        cov_gene, agg_gene = computeGeneAgg(args.predfile, tssposs, args.window)
        
        scoreGenebyTss(agg_gene, cov_gene, tssAgg, args.outpath, args.prefix)

        genetsspos = args.outpath+f'{args.prefix}_gene_tss_score_coverage_window{int(len(tssAgg))-1}.tsv'
        plotGeneLoci(args.predfile, genetsspos, args.window, args.outpath, args.prefix, args.gtfFile, args.ncluster, args.min_cor, args.min_cov)
    
    else:
        plotGeneLoci(args.predfile, args.genetsspos, args.window, args.outpath, args.prefix, args.gtfFile, args.ncluster, args.min_cor, args.min_cov, args.crange)