from tqdm import tqdm
import os
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


def scoreGenebyTss(agg_gene, cov_gene, tssAgg, window, outpath, prefix):
    outfile = open(outpath+f'{prefix}_well_positioned_genes_score_coverage_window{int(len(tssAgg))-1}.tsv', 'w')
    outfile.write('geneid\tspearman_cor\tpearson_cor\tpval\tcoverage\tvariation\tchr\ttss\tstrand\n')
    outf_cov = open(os.path.join(outpath,  prefix+f'_gene_tss_{window}_coverage.bed'), 'w')
    outf_scores = open(os.path.join(outpath,  prefix+f'_gene_tss_{window}_scores.bed'), 'w')
    for chrom in agg_gene:
        for gene in tqdm(agg_gene[chrom]):
            outf_cov.write(f'{chrom}\t{gene[0]}\t{gene[2]}\t{gene[4]}\t{cov_gene[chrom][gene]}\t{gene[3]}\n')
            agg_score_mean = ','.join([str(round(np.mean(x), 3)) if len(x) > 0 else '0' for x in agg_gene[chrom][gene]])
            agg_score_std = ','.join([str(round(np.var(x), 3)) if len(x) > 0 else '0' for x in agg_gene[chrom][gene]])
            outf_scores.write(f'{chrom}\t{gene[0]}\t{gene[2]}\t{gene[4]}\t{gene[3]}\t{agg_score_mean}\t{agg_score_std}\n')
            agg_scores = [np.mean(x) if len(x) > 0 else 0 for x in agg_gene[chrom][gene]]
            # no overlapping reads for this gene
            if np.sum(agg_scores) == 0: continue
            res_spear = stats.spearmanr(tssAgg, agg_scores)
            res_pearson = stats.pearsonr(tssAgg, agg_scores)
            spear_cor = res_spear.statistic
            pear_cor = res_pearson.statistic
            pval = round(res_pearson.pvalue, 3)
            cov = cov_gene[chrom][gene]
            var = round(np.mean([np.var(x) if len(x) > 0 else 0 for x in agg_gene[chrom][gene]]), 0)
            outfile.write(f'{gene[4]}\t{spear_cor}\t{pear_cor}\t{pval}\t{cov}\t{var}\t{chrom}\t{gene[1]}\t{gene[3]}\n')
    outfile.close()
    outf_cov.close()
    outf_scores.close()

def plotGeneStats(genetsspos, outpath, prefix):

    generank = pd.read_table(genetsspos)
    generank = generank.sort_values(by='pearson_cor', ascending=False)
    
    
    plt.hist(generank['spearman_cor'], bins = 'auto', density=True)
    plt.xlabel('gene spearman correlation with well positioned +1 nuc')
    plt.ylabel('density')
    plt.savefig(outpath+f'{prefix}_gene_tss_spearman.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.hist(generank['pearson_cor'], bins = 'auto', density=True)
    plt.xlabel('gene pearson correlation with well positioned +1 nuc')
    plt.ylabel('density')
    plt.savefig(outpath+f'{prefix}_gene_tss_spearman.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.hist(generank['coverage'], bins = 'auto', density=True)
    plt.xlabel('gene coverage')
    plt.ylabel('density')
    plt.savefig(outpath+f'{prefix}_gene_tss_coverage.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.hist(generank['variation'], bins = 'auto', density=True)
    plt.xlabel('heterogeneity per gene')
    plt.ylabel('density')
    plt.savefig(outpath+f'{prefix}_gene_tss_heterogeneity.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

def filterGeneList(predfile, genetsspos, window, outpath, prefix, min_cor=0.6, min_cov=20, max_var=8000, plot_loci = False, gtfFile='', ncluster=''):
    generank = pd.read_table(genetsspos)
    generank
    generank = generank.sort_values(by='pearson_cor', ascending=False)
    
    generank_wellpos = generank[(generank['pearson_cor'] >= min_cor) & (generank['coverage'] >= min_cov) & (generank['variation'] <= max_var)]
    print(f'number of genes with good coverage and well positioned tss: {generank_wellpos.shape[0]}')
    
    outf = open(os.path.join(outpath,  prefix+f'_well_positioned_genes_cov{min_cov}_cor{min_cor}_var{max_var}.bed'), 'w')
    hw = int(window/2)
    genes = []
    regions = []
    for i in range(generank_wellpos.shape[0]):
        geneid= generank_wellpos.iloc[i]['geneid']
        genes.append(geneid)
        chrom = generank_wellpos.iloc[i]['chr']
        tss = int(generank_wellpos.iloc[i]['tss'])
        strand = generank_wellpos.iloc[i]['strand']
        cor = round(float(generank_wellpos.iloc[i]['pearson_cor']), 3)
        outf.write(f'{chrom}\t{tss-hw}\t{tss+hw}\t{geneid}\t{cor}\t{strand}\n')
        pregion = f'{chrom}:{tss-hw}-{tss+hw}'
        regions.append(pregion)
        if plot_loci:
            print(f'gene: {geneid}...')
            plotModTrack(predfile, pregion, ncluster=ncluster, outpath=outpath, prefix= f'{prefix}_{geneid}', gtfFile=gtfFile, xticks_space = 150, na_thred=0.2)
    outf.close()
    return genes, regions

def add_parser(parser):
    parser.add_argument('--predfile', type = str, default='', help = 'prediction output file.')
    parser.add_argument('--nuc_agg', type = str, default='', help = 'aggregated scores at +1 nuc')
    parser.add_argument('--bed', type = str, default='', help = '+1 nucleosome positions in bed format.')
    parser.add_argument('--ref', type = str, default='', help = 'reference genome.')
    parser.add_argument('--window', type = int, default=600, help = 'window size around +1 nuc to compute spearman correlation.')
    parser.add_argument('--inwindow', type = int, default=2000, help = 'input +1 nuc aggregation file window size')
    parser.add_argument('--crange', nargs='+', type = int, default=[20,120], help = 'color range for plotting modification tracks.')
    parser.add_argument('--outpath', type = str, default='', help = 'output path.')
    parser.add_argument('--prefix', type = str, default='', help = 'outfile prefix.')
    parser.add_argument('--genetsspos', type = str, default='', help = 'gene tss nuc positioning score and coverage file. ')
    parser.add_argument('--gtfFile', type = str, default='', help = 'gene annotation file')
    parser.add_argument('--ncluster', type = int, default=1, help = 'number of clusters for clustering reads.')
    parser.add_argument('--plot_loci', action = 'store_true', help = 'plot single loci track on well positioned genes.')
    parser.add_argument('--min_cor', type = float, default=0.6, help = 'minimum pearson correlation to keep for genes to plot.')
    parser.add_argument('--min_cov', type = int, default=15, help = 'minimum coverage to keep for genes to plot.')
    parser.add_argument('--max_var', type = int, default=6000, help = 'maximum variation between reads for each gene.')


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
        plotGeneStats(genetsspos, args.outpath, args.prefix)
        
        getGeneList(args.predfile, genetsspos, args.window, args.outpath, args.prefix, min_cor=args.min_cor, min_cov=args.min_cov, max_var=args.max_var, plot_loci=False, gtfFile=args.gtfFile, ncluster=args.ncluster)
    
    else:
        plotGeneStats(args.genetsspos, args.outpath, args.prefix)
        
        getGeneList(args.predfile, args.genetsspos, args.window, args.outpath, args.prefix, min_cor=args.min_cor, min_cov=args.min_cov, max_var=args.max_var, plot_loci=args.plot_loci, gtfFile=args.gtfFile, ncluster=args.ncluster)