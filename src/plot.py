import os
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import numpy as np
import pysam
from bisect import bisect_left
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from predict import aggregate_scores
from seqUtil import fetchSize

def gettss(bed, genome, window, col = {'chrom':0, 'start':1, 'end':2, 'strand':5}):
    
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
                pos = int(line[col['start']]) if dir == '+' else int(line[col['end']])
                if chr not in tssposs:
                    tssposs[chr] = []
                left=int(max(0, pos-hw))
                right=int(min(pos+hw, genomeSize[chr]))
                tssposs[chr].append((left, pos, right, dir))
        return tssposs

def plotmetagene(predout, bed, genome, window, method, sw='', space=150, labels=('distance to tss (bp)', 'prediction score'), 
                 thred = '', outpath='', prefix = '', color = 'tab:blue', legend = '', odd=False, ylim = (0,1), alpha=0.4, 
                 return_value=False, bed_col = {'chrom':0, 'start':1, 'end':2, 'strand':5}, strand = ''):
    
    tssposs = gettss(bed=bed, genome=genome, window=window, col = bed_col)

    hw = window/2
    all_tss_scores = []
    
    if not isinstance(predout, list):
        predout = [predout]
        color = [color]
        legend = [legend]
    for i in range(len(predout)):
        tssscores = [[] for i in range(window+1)]
        pred = predout[i]
        for chrom, read_strands in pred.items():
            if chrom not in tssposs:
                print(chrom, ' not in input bed.')
                continue
            # for each read
            for read_strand, read_pred in tqdm(read_strands.items()):
                if strand:
                    if read_strand[1] != strand:
                        continue
                if not read_pred:
                    continue
                sortedread = sorted(read_pred.items())
                readStart, readEnd = sortedread[0][0], sortedread[-1][0]
                # tsspos is a set with items (left, pos, right, dir)
                for tsspos in tssposs[chrom]:
                    # no more overlaping tsspos
                    if readEnd < tsspos[0]:
                        break
                    # no everlap with this tsspos
                    if readStart > tsspos[2]:
                        continue
                    # read overlaps with this tsspos
                    for (pos, scores) in sortedread:
                        # last pos falls in the window
                        if pos > tsspos[2]:
                            break
                        if pos < tsspos[0]:
                            continue
                        repos = int(hw+pos-tsspos[1]) if tsspos[3] == '+' else int(tsspos[1]-pos+hw)
                        score = aggregate_scores(scores, method[0])
                        if odd:
                            score = 0.99 if score == 1 else score
                            score = np.log(score/(1-score))
                        tssscores[repos].append(score)
        
        
        tssscores = [aggregate_scores(x, method[1], thred) if len(x) > 0 else 0 for x in tssscores]
        all_tss_scores.append(tssscores)
        print(len(all_tss_scores))
    

    plt.figure(figsize=(6,4))
    for i in range(len(all_tss_scores)):
        tssscores = all_tss_scores[i]
        print(len(tssscores))
        if sw:
            hsw = int(round(sw/2))
            xval, yval = [], []
            for j in range(hsw, (window+1)-(hsw+1), int(round(hsw/2))):
                thesescores = tssscores[j-hsw:j+hsw]
                avg = sum(thesescores)/len(thesescores)
                yval.append(avg)
                xval.append(j-hw)
        else:
            yval = tssscores
            xval = np.arange(-hw, hw+1)
        plt.plot(xval, yval, color=color[i], label=legend[i], alpha=alpha)
    
    plt.xticks(np.concatenate((np.flip(np.arange(0, -hw-1, -space)[1:]), np.arange(0, hw+1, space)), axis=0), rotation='vertical')
    plt.grid(alpha=0.5,axis = 'x')
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    plt.savefig(outpath+prefix+'_aggregate.pdf', bbox_inches='tight', dpi = 200)
    plt.show()
    plt.close()
    if return_value:
        return all_tss_scores

def plotDistribution(predout, outpath, prefix, method, legend, color):
    
    if not isinstance(predout, list):
        predout = [predout]
        color = [color]
        legend = [legend]

    preds = []
    for i in range(len(predout)):
        pred = []
        for chrom, reads in predout[i].items():
            for read, read_pred in reads.items():
                for pos, scores in read_pred.items():
                    pred.append(aggregate_scores(scores, method[0]))
        print('mean: ', np.mean(pred))
        preds.append(pred)
    
    for i in range(len(preds)):
        plt.hist(preds[i], bins=100, alpha=0.4, label = legend[i], color=color[i])
    
    plt.title(prefix)
    plt.legend()
    plt.savefig(outpath+prefix+'_dist.pdf', bbox_inches='tight',dpi = 200)
    plt.show()
    plt.close()


def bedtoPred(bed):
    pred={}
    with open(bed) as infile:
        for line in infile:
            line=line.strip().split('\t')
            chr = line[0]
            if chr not in pred:
                pred[chr] = {'read':{}}
            astart = int(line[1])
            prob = float(line[3])
            pred[chr]['read'][astart] = prob
    return pred

def plotModTrack(ax, labels, mtx, readnames, strands, pregion, cutoff = 0.5,
                 prefix = '', outpath = '', bottom = 0, height = 1, line_width = 0, agg_adjust = 0.4,
                 label = '', colorPalette = 'viridis', xticks_space = 100, ylim_adjust = 0):


    
        
    chrom = pregion.split(':')[0]
    locus = pregion.split(':')[1].split('-')
    pstart, pend = int(locus[0]), int(locus[1])
    
    clustered_idx = [x for _, x in sorted(zip(labels, np.arange(0,len(readnames))))]
    thiscluster = ''

    total, count = np.zeros(mtx.shape[1], dtype = float), np.zeros(mtx.shape[1], dtype = int)
    
    tick_yaxis, label_yaxis = [],[]
    tick_clusters, label_clusters = [],[]

    (R,G,B) = colorMap(palette = colorPalette)
    extend = len(readnames)*agg_adjust
    
    print('plotting genome track with modifications...')
    for i in tqdm(clustered_idx):

        left = pstart
                
        if label == 'strand':
            tick_yaxis.append(bottom)
            symbol = '+' if strands[i] == 1 else '-'
            label_yaxis.append(symbol)
        
        elif label == 'readname':
            tick_yaxis.append(bottom)
            label_yaxis.append(str(readnames[i]))
        
        if labels[i] != thiscluster:
            
            label_clusters.append('c'+str(labels[i]))
            thiscluster = labels[i]
            
            if thiscluster:
                aggregate = count/total
                if np.max(total) < 3:
                    aggregate = np.zeros(mtx.shape[1])
                # aggregate = ((aggregate-np.min(aggregate))/(np.max(aggregate)-np.min(aggregate)))
                for j in range(mtx.shape[1]):
                    rectangle = mplpatches.Rectangle([left, bottom-height*0.5], 1, aggregate[j]*extend, 
                                                     facecolor = 'lightskyblue', edgecolor = 'black', linewidth = 0)
                    ax.add_patch(rectangle)
                    left += 1
                left = pstart
                total, count = np.zeros(mtx.shape[1], dtype = float), np.zeros(mtx.shape[1], dtype = int)
                bottom +=(np.max(aggregate)*extend+1)
            
            tick_clusters.append(bottom)
            
        for j in range(mtx.shape[1]):
            score = mtx[i, j]

            # compute sum of binarized scores across reads in this cluster
            if score >= (cutoff*256):
                count[j] += 1
            total[j] += 1
            
            # no score at this position
            if np.isnan(score):
                col = 'lightgray'
            else:
                color = int((score/256)*100)
                col=(R[color],G[color],B[color])
            rectangle = mplpatches.Rectangle([left, bottom-(height*0.5)], 1, height, 
                                             facecolor = col, edgecolor = 'silver', linewidth = line_width)
            ax.add_patch(rectangle)
            left += 1
        bottom +=height
    
    aggregate = count/total
    
    if np.max(total) < 3:
        aggregate = np.zeros(mtx.shape[1])
    left = pstart

    # add the last aggregated prediction score for this cluster
    for j in range(mtx.shape[1]):
        rectangle = mplpatches.Rectangle([left, bottom-height*0.5], 1, aggregate[j]*extend, facecolor = 'lightskyblue', edgecolor = 'black', linewidth = 0)
        ax.add_patch(rectangle)
        left += 1
    
    bottom +=(np.max(aggregate)*extend+1)
    
    ax.set_xlim(pstart, pend)
    ax.set_ylim(-1.5, bottom)

    ax.tick_params(
        bottom=True, labelbottom=True,
        left=False, labelleft=True,
        right=False, labelright=False,
        top=False, labeltop=False)
    
    ax.set_yticks(ticks= tick_clusters, labels = label_clusters)
    if label in ['readname', 'strand']:
            ax.set_yticks(ticks= tick_yaxis, labels = label_yaxis)
    
    ax.set_xticks(ticks= np.arange(pstart, pend+1, xticks_space))
    ax.set_xticklabels(ax.get_xticks(), rotation = 50)


def colorMap(palette):
    if palette == 'viridis':
        viridis5 = (253/255, 231/255, 37/255)
        viridis4 = (94/255, 201/255, 98/255)
        viridis3 = (33/255, 145/255, 140/255)
        viridis2 = (59/255, 82/255, 139/255)
        viridis1 = (68/255, 1/255, 84/255)
        R1=np.linspace(viridis1[0],viridis2[0],26)
        G1=np.linspace(viridis1[1],viridis2[1],26)
        B1=np.linspace(viridis1[2],viridis2[2],26)
        R2=np.linspace(viridis2[0],viridis3[0],26)
        G2=np.linspace(viridis2[1],viridis3[1],26)
        B2=np.linspace(viridis2[2],viridis3[2],26)
        R3=np.linspace(viridis3[0],viridis4[0],26)
        G3=np.linspace(viridis3[1],viridis4[1],26)
        B3=np.linspace(viridis3[2],viridis4[2],26)
        R4=np.linspace(viridis4[0],viridis5[0],26)
        G4=np.linspace(viridis4[1],viridis5[1],26)
        B4=np.linspace(viridis4[2],viridis5[2],26)
        R=np.concatenate((R1[:-1],R2[:-1],R3[:-1],R4),axis=None)
        G=np.concatenate((G1[:-1],G2[:-1],G3[:-1],G4),axis=None)
        B=np.concatenate((B1[:-1],B2[:-1],B3[:-1],B4),axis=None)
        return (R,G,B)
    
    elif palette == 'plasma':
        plasma5 = (237/255, 252/255, 27/255)
        plasma4 = (245/255, 135/255, 48/255)
        plasma3 = (190/255, 48/255, 101/255)
        plasma2 = (87/255, 0/255, 151/255)
        plasma1 = (15/255, 0/255, 118/255)
        plasma = [plasma1, plasma2, plasma3, plasma4, plasma5]
        
        colorCode = {'R': 0 , 'G': 1, 'B' : 2}
        myRGB = {'R':[] , 'G': [], 'B': []}
        
        for i in range(len(plasma)-1):
            step =25
            if i == 3:
                step =26
            for code in colorCode.keys():
                col = np.linspace(plasma[i][colorCode[code]], plasma[i+1][colorCode[code]], step)
                myRGB[code].extend(col)
        return (myRGB['R'],myRGB['G'],myRGB['B'])


def predToMtx(pred_dict, pregion, outpath = '', prefix = '', impute = False, strand = '',
              strategy = 'most_frequent', filter_read = True, write_out = True, na_thred = 0.5):

    '''
    predToMtx function formats input prediction tsv file into an matrix.
    
    input:
        prediction file
        outpath
        prefix
        region
        pregion
        step
        inpute
        strategy
        filter_read
    output:
        output matrix file
    return:
        readnames
        strands
    '''

    chrom = pregion.split(':')[0]
    locus = pregion.split(':')[1].split('-')
    pstart, pend = int(locus[0]), int(locus[1])
    outfile = outpath + prefix + '_' + pregion + '.mtx'

    mtx = []
    readnames = []
    strands = []
    
    
    for thischrom in pred_dict:
         # filter by chromsome
        if thischrom != chrom:
            continue
        for (thisread, thisstrand) in pred_dict[thischrom]:
            # filter by strand
            if strand:
                if thisstrand != strand:
                    continue
            sortedread = sorted(pred_dict[thischrom][(thisread, thisstrand)].items())
            # filter by positions
            if sortedread[0][0] > pend or sortedread[-1][0] < pstart:
                continue
            # store score for each pos
            pos_scores = {i:-1 for i in range(pstart, pend+1)}
            poss = [i[0] for i in sortedread]
            
            left = bisect_left(poss, pstart)
            
            for i in range(left, len(sortedread)):
                pos, scores = sortedread[i]
                if pos > pend:
                    break
                if pos not in pos_scores:
                    continue
                pos_scores[pos] = np.mean(scores)
            thisscores = [i for i in pos_scores.values()]
            if np.sum(thisscores) != -1*len(thisscores):
                mtx.append(thisscores)
                readnames.append(thisread)
                strands.append(thisstrand)
    
    mtx = np.array(mtx, dtype = float)
    mtx[mtx==-1] = np.nan
    readnames = np.array(readnames, dtype = str)
    strands = np.array(strands, dtype = int)

    if filter_read:
        print('number of reads before filtering:', len(readnames))
        little_na = np.invert(np.isnan(mtx).sum(axis = 1)>(mtx.shape[1]*na_thred))
        mtx = mtx[little_na,:]
        readnames = readnames[little_na]
        strands = strands[little_na]
        print('number of reads kept:', len(readnames))

    if impute:
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        mtx = imp.fit_transform(mtx)
    
    if write_out:
        print('writing output to file: ', outfile)
        mtxFh = open(outfile, 'w')
        for line in mtx:
            mtxFh.write(','.join(np.array(line, dtype = str)) + '\n')
        mtxFh.close()
    
    if np.isnan(mtx).sum() != 0:
        print('nan in output matrix!')

    return np.array(mtx), np.array(readnames), np.array(strands)


def clusterRead(pred_dict, pregion, outpath, prefix, n_cluster = '', random_state = 42, method = '', show_elbow = False, nPC= 5, na_thred = 0.5, strand = '', strategy='most_frequent'):
    '''
    
    ClusterRead function takes a modification prediction file as input and perform kmeans clustering on reads.
    
    input:
        predict: modification prediction tsv generated from modPredict function.
        n_cluster: number of centroids in kmeans clustering. If not assigned, number of cluster will be chosen based on the largest silhouette score among number of clusters 2-6.
        random_state: set seed to replicate the results
        method:
            pca: run pca on prediction matrix and use the first nPC to perform kmean clustering. NA values are imputed by the most frequent value in prediction matrix.
            cor: run spearman correlation on prediction matrix, and perform kmean clustering on spearman distances. NA values are kept in prediction matrix, but is ommited in spearman correlation analysis.
            default(''): run kmean clustering on prediction matrix. NA values are imputed by the most frequent value in prediction matrix.
        nPC: number of principal components to use in clustering if method is set as 'pca'.
        na_thred: percent of missing positions allowed each read. The less this value is, the more stricter filtering is.
    output:
        outpath: output file path
        prefix: output file prefix
    return:
        readnames, strands, mtx, bins
    '''
    
    print('preprocessing input matrix...')
    
    # perform dimension reduction with pca before clustering
    if method == 'pca':
        print('Reading prediction file and outputing matrix...')
        prefix = prefix + "_method_pca"
        mtx, readnames, strands = predToMtx(pred_dict=pred_dict, pregion=pregion, outpath=outpath, prefix=prefix, na_thred=na_thred, strand=strand)
        if np.isnan(mtx).sum() != 0:
            imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
            new_mtx = imp.fit_transform(mtx)
        
        print('running pca...')
        pca = PCA(n_components=nPC)
        new_mtx = pca.fit(new_mtx).transform(new_mtx)
        
        rel_height = 1.2
        rel_width = 1.2
        lw = 0
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5 * rel_width, 5 * rel_height))
        
        ax1.scatter(new_mtx[:, 0], new_mtx[:, 1], color='orange', alpha=1, lw=lw)
        ax1.set_xlabel('PC0')
        ax1.set_ylabel('PC1')
        
        ax2.scatter(new_mtx[:, 1], new_mtx[:, 2], color='orange', alpha=1, lw=lw)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        
        ax3.scatter(new_mtx[:, 2], new_mtx[:, 3], color='orange', alpha=1, lw=lw)
        ax3.set_xlabel('PC2')
        ax3.set_ylabel('PC3')
        
        ax4.scatter(new_mtx[:, 3], new_mtx[:, 4], color='orange', alpha=1, lw=lw)
        ax4.set_xlabel('PC3')
        ax4.set_ylabel('PC4')
        plt.subplots_adjust(wspace=0.35, hspace=0.35)
        plt.savefig(outpath + prefix + pregion + "_pca.pdf", bbox_inches='tight')
        plt.close()
        
    # perform pairwise spearmanr correlation analysis before clustering
    elif method == 'cor':
        print('Reading prediction file and outputing matrix...')
        prefix = prefix + "_method_cor"
        mtx, readnames, strands = predToMtx(pred_dict=pred_dict, pregion=pregion, outpath=outpath, prefix=prefix, na_thred=na_thred, strand=strand)
        res = stats.spearmanr(mtx, axis = 1, nan_policy = 'omit')
        new_mtx = res.statistic

    else:
        mtx, readnames, strands = predToMtx(pred_dict=pred_dict, pregion=pregion, outpath=outpath,  prefix=prefix, na_thred=na_thred, strand=strand)
        if np.isnan(mtx).sum() != 0:
            imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
            new_mtx = imp.fit_transform(mtx)

    try:
        assert len(readnames) == mtx.shape[0]
    except:
        print('dimentions do not match!')
        print('length of readnames:', len(readnames))
        print('length of strands:', len(strands))
        print('matrix dimension:', mtx.shape)

    # select the best number of clusters based on silhouette score
    if not n_cluster:
        inertias = []
        silhouette_avgs = []
        n_clusters = [2, 3, 4, 5, 6]
        for n_cluster in n_clusters:
    
            clusterer = KMeans(n_clusters=n_cluster, random_state=random_state)
            cluster_labels = clusterer.fit_predict(new_mtx)
            inertias.append(clusterer.inertia_)
            silhouette_avg = silhouette_score(new_mtx, cluster_labels)
            silhouette_avgs.append(silhouette_avg)
            print(
                "For n_clusters =",
                n_cluster,
                "The average silhouette_score is :",
                silhouette_avg,
            )
            
        if show_elbow:
            plt.plot(n_clusters, inertias, marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.close()
        
        n_cluster = n_clusters[np.argmax(silhouette_avgs)]
    
    print('Clustering with number of clusters =', n_cluster)
    outfile = open(outpath + prefix + 'clustering.tsv', 'w')
    outfile.write('readname\tstrand\tcluster\n')
    
    # perform kmeans clustering
    kmeans = KMeans(
        init="random",
        n_clusters=n_cluster,
        n_init=10,
        max_iter=300,
        random_state=random_state)

    kmeans.fit(new_mtx)
    for r,s,l in zip(readnames, strands, kmeans.labels_):
        line = '{}\t{}\t{}\n'.format(r, s, l)
        outfile.write(line)
    outfile.close()

    return kmeans.labels_, mtx, readnames, strands


def readGTF(gtfFile, chromPlot, startPlot, endPlot, genePlot, geneSlot, features):

    gtfReads = {}
    gene = ''
    count = 0

    with open(gtfFile) as gtfFh:
        for line in gtfFh:
            if '#!' in line:
                continue
            else:
                line = line.split('\t')
                chrom = str(line[0])
                start = int(line[3])
                end = int(line[4])
                feature = line[2]
                if chrom != chromPlot:
                    continue
                else:
                    if feature in features:
                        splitPoint = str(genePlot[feature]) + ' "'
                        transcript = line[8].split(';')[geneSlot[feature]]
                        if splitPoint not in transcript:
                            continue
                        geneID = transcript.split(splitPoint)[1].split('"')[0]
                        # New gene
                        if geneID != gene:
                        # Store the previous genecript
                            if gene in gtfReads:
                                minstart = min(gtfReads[gene]['starts'])
                                maxend = max(gtfReads[gene]['ends'])
                                if maxend <= startPlot:
                                     del gtfReads[gene]
                                elif minstart >= endPlot:
                                     del gtfReads[gene]
                                else:
                                    gtfReads[gene]['start'] = minstart
                                    gtfReads[gene]['end'] = maxend
                                    # print(gtfReads[gene]['start'])
                                    count +=1
                                    # print(count)
                            gene = geneID
                            gtfReads[gene] = {'starts': [start],'ends': [end]}
                            gtfReads[gene][feature] = ([start], [end])
                        else:
                            gtfReads[gene]['starts'].append(start)
                            gtfReads[gene]['ends'].append(end)
                            if feature in gtfReads[gene]:
                                gtfReads[gene][feature][0].append(start)
                                gtfReads[gene][feature][1].append(end)
                            else:
                                gtfReads[gene][feature] = ([start], [end])
        if gene in gtfReads:
            minstart = min(gtfReads[gene]['starts'])
            maxend = max(gtfReads[gene]['ends'])
            if maxend <= startPlot:
                    del gtfReads[gene]
            elif minstart >= endPlot:
                    del gtfReads[gene]
            else:
                gtfReads[gene]['start'] = minstart
                gtfReads[gene]['end'] = maxend

    sorted_gtfReads = dict(sorted(gtfReads.items(), key = lambda x:x[1]['start']))
    return (features, sorted_gtfReads)

def plotGtfTrack(plot, gtfFile, region, features = ['CDS', 'start_codon'], genePlot = {'CDS': 'gene_name', 'start_codon': 'gene_name'}, 
                 geneSlot = {'CDS': 3, 'start_codon': 3}, adjust_features = '', label_name = True, label_direction = False, 
                 colorpalates= ['royalblue', 'darkorange'], thinHeight = 0.2, Height = [0.8, 0.8], line_width = 0):
    
    
    chrom = region.split(':')[0]
    locus = region.split(':')[1].split('-')
    startPlot, endPlot = int(locus[0]), int(locus[1])
    prange = endPlot-startPlot
    if not adjust_features:
        adjust_features = [0, 0]
        adjust_features[1] = prange/100
    features, sorted_gtfReads = readGTF(gtfFile, chromPlot = chrom.split('chr')[1], 
                                        genePlot = genePlot, geneSlot = geneSlot, 
                                        startPlot = startPlot, endPlot = endPlot, features = features)
    
    print('plotting gene annotations...')
    
    yRightMost = {}
    
    for transID in sorted_gtfReads:
        y = 0
        start = sorted_gtfReads[transID]['start']
        end = sorted_gtfReads[transID]['end']
        while True:
            if y not in yRightMost:
                bottom = y
                yRightMost[y] = end
                break
            elif start >= yRightMost[y]+1500:
                bottom = y
                yRightMost[y] = end
                break
            else:
                y +=1
        rectangle = mplpatches.Rectangle([start, bottom-(thinHeight/2)], end - start, thinHeight,
                                        facecolor = 'grey',
                                        edgecolor = 'black',
                                        linewidth = line_width)
        if label_name:
            textStart = start
            if start < startPlot:
                textStart = startPlot
            plot.text(x = textStart-1, y = bottom, s = transID, ha = 'right', va = 'center', size = 'small')
            
        plot.add_patch(rectangle)
        
        # plot feature1
        if features[0] in sorted_gtfReads[transID]:
            blockStarts = np.array(sorted_gtfReads[transID][features[0]][0], dtype  = int)
            blockEnds = np.array(sorted_gtfReads[transID][features[0]][1], dtype = int)
            for index in range(len(blockStarts)):
                blockStart = blockStarts[index]
                blockEnd = blockEnds[index]
                rectangle = mplpatches.Rectangle([blockStart-adjust_features[0], bottom-(Height[0]/2)],
                                                 blockEnd-blockStart+adjust_features[0], Height[0],
                                    facecolor = colorpalates[0],
                                    edgecolor = 'black',
                                    linewidth = line_width)
                plot.add_patch(rectangle)
        
        # plot feature2
        if features[1] in sorted_gtfReads[transID]:
            blockStarts = np.array(sorted_gtfReads[transID][features[1]][0], dtype = int)
            blockEnds = np.array(sorted_gtfReads[transID][features[1]][1], dtype = int)
            for index in range(0, len(blockStarts), 1):
                blockStart = blockStarts[index]
                blockEnd = blockEnds[index]
                rectangle = mplpatches.Rectangle([blockStart-adjust_features[1], bottom-(Height[1]/2)], 
                                                 blockEnd-blockStart+adjust_features[1], Height[1],
                                    facecolor = colorpalates[1],
                                    edgecolor = 'black',
                                    linewidth = line_width)
                plot.add_patch(rectangle)
    
    plot.set_xlim(startPlot, endPlot)
    plot.set_ylim(-1,3)
    plot.tick_params(bottom=False, labelbottom=False,
                   left=False, labelleft=False,
                   right=False, labelright=False,
                   top=False, labeltop=False)

def plotSingleRead(pred_dict, pregion, outpath, prefix):

    chrom = pregion.split(':')[0]
    locus = pregion.split(':')[1].split('-')
    pstart, pend = int(locus[0]), int(locus[1])
    
    c = 0
    for read in tqdm(pred_dict[chrom]):
        readID = read[0]
        strand = read[1]
        thiscolor = 'red' if strand == 1 else 'blue'
        sortedread = sorted(pred_dict[chrom][read].items())
        poss = [i[0] for i in sortedread]
        scores = [np.mean(i[1]) for i in sortedread]
        plt.plot(poss, scores, ms= 0.5, alpha = 0.5, color = thiscolor, label = f'{strand}')
        if c == 0:
            plt.legend()
        c+=1
    plt.xlim(pstart, pend)
    plt.xlabel('genomic position (bp)')
    plt.ylabel('prediction score')
    outfig = os.path.join(outpath, prefix+'_score_per_pos_by_read_col_by_strand.pdf')
    plt.savefig(outfig, bbox_inches='tight')


def plotAggregatedScore(pred_dicts, pregion, outpath, prefix, labels, fmt='-', mnase = ''):
    
    chrom = pregion.split(':')[0]
    locus = pregion.split(':')[1].split('-')
    pstart, pend = int(locus[0]), int(locus[1])

    for i in range(len(pred_dicts)):
        pred_dict = pred_dicts[i]
        agg_scores = {}
        for read in tqdm(pred_dict[chrom]):
            readID = read[0]
            strand = read[1]
            for pos in pred_dict[chrom][read]:
                if pos not in agg_scores:
                    agg_scores[pos] = [np.mean(pred_dict[chrom][read][pos])]
                else:
                    agg_scores[pos].append(np.mean(pred_dict[chrom][read][pos]))

        agg_scores = sorted(agg_scores.items())
        poss = [i[0] for i in agg_scores]
        scores = [np.mean(i[1]) for i in agg_scores]
        
        plt.plot(poss, scores, fmt, ms= 0.5, alpha = 0.6, label = labels[i])

    if mnase:
        poss = []
        mnase_scores = []
        with open(mnase, 'r') as infile:
            for line in infile:
                line = line.strip().split('\t')
                chr = line[0]
                if chr != chrom:
                    continue
                pos = int(line[1])
                if pos < pstart or pos > pend:
                    continue
                score = (float(line[3])/1400)*256
                poss.append(pos)
                mnase_scores.append(score)
        
        plt.plot(poss, mnase_scores, fmt, ms= 0.5, alpha = 0.8, label = 'mnase')
    
    
    plt.xlim(pstart, pend)
    plt.xlabel(f'genomic position (bp)')
    plt.ylabel('prediction score')
    plt.legend()
    outfig = os.path.join(outpath, prefix+'_score_per_pos_aggregated.pdf')
    plt.savefig(outfig, bbox_inches='tight')

def plotMotif(genome, bed, extend=10, shift=-150, center_name = 'tss'):
    '''
    plot motif enrichment given a input bed file and reference genome.
    '''

    def reverseCompliment(seq):
        basePair = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
        rseq = ''
        for i in range(1, len(seq)+1):
            rseq += basePair[seq[-i]]
        return(rseq)
    
        
    refDict = {}
    sequence=''
    with open(genome, 'r') as refFile:
        for line in refFile:
            if line.startswith('>'):
                if sequence:
                    refDict[name]=sequence
                    sequence=''
                name=str(line[1:].strip().split()[0])
            else:
                sequence+=line.strip()
    
        if sequence:
            refDict[name]=sequence

    motif_dict= {i:{'A':0, 'C':0, 'G':0, 'T':0} for i in np.arange(-extend, extend, 1)}
    nCount=0
    
    with open(bed, 'r') as infile:
        for line in infile:
            line = line.strip().split('\t')
            chrom = line[0]
            strand = line[5]
            nuc_pos = int(line[1])
            if shift:
                center = nuc_pos + shift if strand == '-' else nuc_pos - shift
            else:
                center = nuc_pos
            left = center - int(extend)
            right = center + int(extend)
            sequence = refDict[chrom][left:right]
            if strand == '-':
                sequence = reverseCompliment(sequence)
            nCount +=1
            for i in range(len(sequence)):
                nt = sequence[i]
                pos = i-extend
                motif_dict[pos][nt] +=1

    s = 4
    err = (1/np.log(2))*((s-1)/(2*nCount))
    
    
    figureWidth=5
    figureHeight=2
    
    plt.figure(figsize=(figureWidth,figureHeight))
    
    panelWidth=1.5
    panelHeight=0.5
    
    panel1 = plt.subplot()
    
    panel1.tick_params(bottom=True, labelbottom=True,
                       left=True, labelleft=True,
                       right=False, labelright=False,
                       top=False, labeltop=False)
    
    panel1.set_xlim(-extend,extend)
    
    panel1.set_xticks(np.arange(-extend,extend+1, 5))
    # panel1.set_yticks([0, 1])
    panel1.set_xlabel(f"Distance to\n {center_name}")
    panel1.set_ylabel("Bits")
    panel1.set_title("sacCer3 genome")
    panel1.axvline(x = 0, color = 'black', linewidth = 0.5)
    
    max_y = 0
    
    for pos in motif_dict.keys():
        ntCount = list(motif_dict[pos].values())
        totalCount = sum(ntCount)
        freqCount = [nt/totalCount for nt in ntCount]
        entropy = sum([-p*np.log2(p) for p in freqCount])
        colHeight = np.log2(4) - (entropy+err)
        if colHeight > max_y:
            max_y = colHeight
        height = {nt: freq*colHeight for nt,freq in zip(['A', 'C', 'G', 'T'], freqCount)}
        Sortedheight = dict(sorted(height.items(), key=lambda x:x[1]))
        bottom = 0
        top = 0
        for alphabet in Sortedheight.keys():
            bottom = top
            top = bottom + Sortedheight[alphabet]         #left,right,bottom,top
            panel1.imshow(pngList[alphabet],extent=[pos,pos+1,bottom,top],aspect='auto',origin='upper')

    panel1.set_ylim(0, max_y)