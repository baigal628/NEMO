from seqUtil import *
from bamUtil import *
from nanoUtil import *
from nntUtil import *
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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

def predToMtx(infile, pregion, bins, outpath = '', prefix = '', step = 40, impute = True, 
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
    rstart = int(np.floor((pstart-bins[0])/step))
    rend = int(np.ceil((pend-bins[0])/step))
    
    outfile = outpath + prefix + '_' + pregion + '.mtx'
    readnames = []
    strands = []
    mtx = []
    
    with open(infile, 'r') as predFh:
        for line in predFh:
            bin_scores = np.empty(len(bins), dtype = float) * np.nan
            line = line.strip().split('\t')
            readname = line[0]
            strand = line[1]
            start = int(line[2])
            if start > bins[-1]:
                continue
            probs = line[3].split(',')
            end = start + step*(len(probs)-1)
            if end < bins[0]:
                continue
            else:
                i = int((start-bins[0])/step)
                if i < 0:
                    probs = probs[-i:]
                    i = 0
                for prob in probs:
                    bin_scores[i] = prob
                    i +=1
                    if i >= len(bins):
                        break
            readnames.append(readname)
            strands.append(strand)
            mtx.append(bin_scores)

    mtx = np.array(mtx, dtype = float)  
    mtx = mtx[:,rstart:rend]
    bins = bins[rstart:rend]
    readnames = np.array(readnames, dtype = int)
    strands = np.array(strands, dtype = int)
    
    if filter_read:
        little_na = np.invert(np.isnan(mtx).sum(axis = 1)>(mtx.shape[1]*na_thred))
        mtx = mtx[little_na,:]
        readnames = readnames[little_na]
        strands = strands[little_na]
    
    if impute: 
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        mtx = imp.fit_transform(mtx)
    
    if write_out:
        print('writing output to file: ', outfile)
        mtxFh = open(outfile, 'w')
        mtxFh.write(','.join(np.array(bins, dtype = str)) + '\n')
        for binscore in mtx:
            mtxFh.write(','.join(np.array(binscore, dtype = str)) + '\n')
        mtxFh.close()
    
    if np.isnan(mtx).sum() != 0:
        print('nan in output matrix!')

    return np.array(readnames), np.array(strands), np.array(mtx), bins

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
                print(Height[0])
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

def clusterRead(predict, outpath, prefix, region, pregion, bins, step = 40, n_cluster = '', random_state = 42, method = '', show_elbow = True, nPC= 5, na_thred = 0.5):
    '''
    
    ClusterRead function takes a modification prediction file as input and do clustering on reads.
    
    input:
        predict: modification prediction tsv generated from modPredict function.
        prefix: output file prefix
        outpath: output path
        n_cluster: number of clusters
        random_state
        method
    output:
        outfile: a tsv file with clustering results.
    return:
        cluster_labels
    '''
    
    print('preprocessing input matrix...')
    if method == 'pca':
        print('Reading prediction file and outputing matrix...')
        prefix = prefix + "_method_pca_"
        readnames, strands, mtx, bins = predToMtx(infile=predict, pregion=pregion, bins=bins, outpath=outpath, prefix=prefix, step=step, na_thred=na_thred)
        print('running pca...')
        pca = PCA(n_components=nPC)
        new_mtx = pca.fit(mtx).transform(mtx)
        
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
        plt.savefig(outpath + prefix + region + "_pca.pdf")
        plt.close()
        
    elif method == 'cor':
        print('Reading prediction file and outputing matrix...')
        prefix = prefix + "_method_cor_"
        readnames, strands, mtx, bins = predToMtx(infile=predict, pregion=pregion, bins=bins, outpath=outpath, prefix=prefix, step=step, impute=False, na_thred=na_thred)
        res = stats.spearmanr(mtx, axis = 1, nan_policy = 'omit')
        new_mtx = res.statistic

    else:
        readnames, strands, mtx, bins = predToMtx(infile=predict, pregion=pregion, bins=bins, outpath=outpath, step=step, prefix=prefix, na_thred=na_thred)
        new_mtx = mtx

    try:
        assert len(readnames) == mtx.shape[0]
    except:
        print('dimentions do not match!')
        print('length of readnames:', len(readnames))
        print('length of strands:', len(strands))
        print('matrix dimension:', mtx.shape)

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
        
        n_cluster = n_clusters[np.argmax(silhouette_avgs)]
    
    print('Clustering with number of clusters =', n_cluster)
    outfile = open(outpath + prefix + 'clustering.tsv', 'w')
    outfile.write('readname\tstrand\tcluster\n')
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

    return kmeans.labels_, readnames, strands, mtx, bins

def plotModDistribution(modPredict_pos, modPredict_neg, modPredict_chrom, return_scores = False):
    '''
    plotModDistribution reads positive, negative and chromatin modification scores from modScores.tsv files and plot histograms.
    '''
    
    with open(modPredict_pos, 'r') as modPFh:
        positions = [int(p) for p in modPFh.readline().strip().split('\t')[1].split(',')]       
        pos_scores = {-1:[], 1:[]}
        for line in modPFh:
            line = line.strip().split('\t')
            strand = int(line[2])
            for score in line[5].split(','):
                pos_scores[strand].append(float(score))
    
    with open(modPredict_neg, 'r') as modPFh:
        modPFh.readline()
        neg_scores = {-1:[], 1:[]}
        for line in modPFh:
            line = line.strip().split('\t')
            strand = int(line[2])
            for score in line[5].split(','):
                neg_scores[strand].append(float(score))
    
    with open(modPredict_chrom, 'r') as modPFh:
        modPFh.readline()
        chrom_scores = {-1:[], 1:[]}
        for line in modPFh:
            line = line.strip().split('\t')
            strand = int(line[2])
            for score in line[5].split(','):
                chrom_scores[strand].append(float(score))
    
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex = True, sharey = 'row')
    axs[0, 0].hist(pos_scores[-1], color='lightblue', bins = 30, label = 'pos', alpha=0.7)
    axs[0, 0].hist(neg_scores[-1], color='lightpink', bins = 30, label = 'neg', alpha=0.7)
    axs[0, 0].set_title('reverse strand cotrol')
    axs[0, 0].legend(prop ={'size': 10})
    axs[0, 1].hist(pos_scores[1], color='lightblue', bins = 30, label = 'pos', alpha=0.7)
    axs[0, 1].hist(neg_scores[1], color='lightpink', bins = 30, label = 'neg', alpha=0.7)
    axs[0, 1].set_title('forward strand control')

    axs[1, 0].hist(chrom_scores[-1], color='orange', bins = 30,)
    axs[1, 0].set_title('reverse strand chromatin')

    axs[1, 1].hist(chrom_scores[1], color='orange', bins = 30,)
    axs[1, 1].set_title('forward strand chromatin')
    
    if return_scores:
        return pos_scores, neg_scores, chrom_scores, positions, fig
    else:
        return fig


def plotModTrack(ax, labels, readnames, strands, mtx, bins, step = 40,
                 prefix = '', outpath = '', bottom = 0, height = 1, line_width = 0, agg_adjust = 0.4,
                 label = '', colorRange = (0.3, 0.5, 0.6), colorPalette = 'viridis', xticks_space = 120, ylim_adjust = 0):

    clustered_idx = [x for _, x in sorted(zip(labels, np.arange(0,len(readnames))))]
    thiscluster = ''

    total, count = np.zeros(len(bins), dtype = float), np.zeros(len(bins), dtype = int)
    
    tick_yaxis, label_yaxis = [],[]
    tick_clusters, label_clusters = [],[]

    (R,G,B) = colorMap(palette = colorPalette)
    extend = len(readnames)*agg_adjust
    
    for i in clustered_idx:
        
        left = bins[0]
        
        if label == 'strand':
            tick_yaxis.append(bottom)
            label_yaxis.append(str(strands[i]))
        elif label == 'readname':
            tick_yaxis.append(bottom)
            label_yaxis.append(str(readnames[i]))
        if labels[i] != thiscluster:
            thiscluster = labels[i]
            label_clusters.append('c'+str(labels[i]))
            if thiscluster:
                count[count==0]=1
                aggregate = total/count
                # aggregate = ((aggregate-np.min(aggregate))/(np.max(aggregate)-np.min(aggregate)))
                for pos in range(len(bins)):
                    rectangle = mplpatches.Rectangle([left, bottom-height*0.5], step, aggregate[pos]*extend,
                                                     facecolor = 'silver', edgecolor = 'black', linewidth = 0.5)
                    ax.add_patch(rectangle)
                    left += step
                
                left = bins[0]
                total, count = np.zeros(len(bins), dtype = float), np.zeros(len(bins), dtype = int)
                bottom +=np.max(aggregate)*extend+1
            tick_clusters.append(bottom)
        
        for pos in range(len(bins)):
            score = mtx[i, pos]
            if np.isnan(score):
                col = 'white'
            else:
                if score >= colorRange[1]:
                    total[pos] += 1
                count[pos] += 1
                color = int(score*100)
                (lower, median, upper) = colorRange
                if color >= upper*100:
                    color = 100
                elif color >= median*100:
                    color = 80
                elif color <= lower*100:
                    color = 0
                col=(R[color],G[color],B[color])
            rectangle = mplpatches.Rectangle([left, bottom-(height*0.5)], step, height, 
                                             facecolor = col, edgecolor = 'silver',
                                             linewidth = line_width)
            ax.add_patch(rectangle)
            left += step
        bottom +=height
    
    count[count==0]=1
    aggregate = total/count
    left = bins[0]
    for pos in range(len(bins)):
        rectangle = mplpatches.Rectangle([left, bottom-height*0.5], step, aggregate[pos]*extend, 
                                         facecolor = 'silver', edgecolor = 'black', linewidth = 0.5)
        ax.add_patch(rectangle)
        left += step

    ax.set_xlim(bins[0], bins[1])
    ax.set_ylim(-1.5,len(readnames)*height+extend*np.max(aggregate)*len(label_clusters)+ylim_adjust)
    
    ax.tick_params(
        bottom=True, labelbottom=True,
        left=False, labelleft=True,
        right=False, labelright=False,
        top=False, labeltop=False)
    
    ax.set_yticks(ticks= tick_clusters, labels = label_clusters)
    ax.set_xticks(ticks= np.arange(bins[0], bins[-1], xticks_space))
    ax.set_xticklabels(ax.get_xticks(), rotation = 50)

# def plotbdgTrack(plot, bdg, region, step = 1, scale = 1000, header = False, col = 'grey', annot = '', ylim = ''):
    
#     chrom = region.split(':')[0]
#     locus = region.split(':')[1].split('-')
#     pstart, pend = int(locus[0]), int(locus[1])
#     ymax = 0
    
#     print('plotting ' , bdg,  '...')
#     with open(bdg, 'r') as bdgFh:
#         if header:
#             header = bdgFh.readlines(1)
#         for line in bdgFh:
#             line = line.strip().split('\t')
#             if line[0] != chrom:
#                 continue
#             start = int(line[1])
#             end =  int(line[2])
#             if end < pstart:
#                 continue
#             elif start > pend:
#                 break
#             else:
#                 prob = float(line[3])
#                 height = min(1.0, prob/scale)
#                 if height > ymax:
#                     ymax = height
#                 left = max(start, pstart)
#                 rectangle = mplpatches.Rectangle([left, 0], end-left, height,
#                         facecolor = col,
#                         edgecolor = 'grey',
#                         linewidth = 0)
#                 plot.add_patch(rectangle)
    
#     plot.set_xlim(pstart, pend)
#     if ylim:
#         plot.set_ylim(0,ylim+0.1)
#     plot.set_ylim(0,ymax+0.1)
#     plot.tick_params(bottom=False, labelbottom=False,
#                    left=False, labelleft=False,
#                    right=False, labelright=False,
#                    top=False, labeltop=False)
#     plot.set_ylabel(annot)
#     print('Finished plotting ' , bdg,  '!')

def plotAllTrack(prediction, gtfFile, bins, region, pregion, na_thred = 0.5, 
                 step = 40, outpath = '', prefix = '', ncluster = 3, method = '', 
                 subset = False, colorPalette = 'viridis', colorRange = (0.3, 0.5, 0.6), vlines = '',
                 gtfFeatures = ['CDS', 'start_codon'],  genePlot = {'CDS': 'gene_name', 'start_codon': 'gene_name'}, 
                 geneSlot = {'CDS': 3, 'start_codon': 3}, gtfHeight = [0.8, 0.8], adjust_features = '',
                 trackHeight = 0.5, fontsize=10, track_ylim_adjust = 0.5, track_agg_adjust = 0.4, xticks_space = 120,
                 fig_size = '', savefig = True, dpi = 800, seed = 42):

    print('Start clustering reads...')
    labels, readnames, strands, mtx, bins = clusterRead(predict=prediction, outpath=outpath, prefix=prefix, 
                                                  region=region, pregion=pregion, bins=bins, step =step, na_thred=na_thred,
                                                  n_cluster=ncluster, method=method, random_state=seed, show_elbow = False)
    print('Finished clustering reads!')

    if subset:
        readIdx = np.arange(len(readnames))
        np.random.shuffle(readIdx)
        readlen = int(np.ceil(len(readIdx)*subset))
        readIdx = readIdx[:readlen]
        labels = labels[readIdx]
        readnames = readnames[readIdx]
        strands = strands[readIdx]
        mtx = mtx[readIdx,]
    if fig_size:
        (figHeight, figWidth) = fig_size
    else:
        figHeight = len(readnames)/8
        figWidth = len(bins)/10
    print('Figure size:', figHeight, figWidth)
    plt.figure(figsize = (figWidth, figHeight))
                #(left, bottom, width, height)
    ax1 = plt.axes((0.1, 0.2 , 0.85, 0.6), frameon=False)
    ax2 = plt.axes((0.1, 0.8, 0.85, 0.16), frameon=False)
    plotModTrack(ax=ax1, labels=labels, readnames=readnames, strands=strands, mtx=mtx, bins=bins, step=step,
                 ylim_adjust=track_ylim_adjust, agg_adjust=track_agg_adjust,
                 colorRange = colorRange, colorPalette=colorPalette, height=trackHeight, xticks_space=xticks_space)
    
    plotGtfTrack(ax2, gtfFile, pregion, Height = gtfHeight, features = gtfFeatures, 
                 genePlot = genePlot, geneSlot = geneSlot, adjust_features=adjust_features)

    if vlines:
        for label, vl in vlines.items():
            plt.axvline(x = vl, color = 'black', linestyle = 'dashed')
            plt.text(vl+12, 0.7, label, fontsize=fontsize)
    
    if savefig:
        outfig = outpath + prefix + '_' + region + '_' + method + '_clustered_reads.pdf'
        plt.savefig(outfig, dpi=dpi)

def plotPredictionScores(scores, modVars, modCounts, labels = ['pos', 'neg', 'chrom']):
    
    strands = [1, -1]
    group = 0
    for pos in range(1,4):
        ax = plt.subplot(4,3,pos)
        strand = strands[0]
        ax.margins(0.05)
        ax.scatter(x = modCounts[group][strand], y = scores[group][strand], 
                   s = 0.8, c = scores[group][strand], label = labels[group])
        ax.set_title(labels[group], size = 'medium')
        if pos ==2:
             ax.set_title('number of modifiable positions (AT/TA) \n neg', size = 'medium')            
        if pos == 1:
            ax.set_ylabel('predicted \n scores (+)')
            ax.tick_params(left = True, labelleft= True,
                           bottom = False, labelbottom = False)
        else:
            ax.tick_params(left = True, labelleft= False,
                           bottom = False, labelbottom = False)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, 30)
        group +=1
    
    group = 0
    
    for pos in range(4, 7):
        ax = plt.subplot(4, 3, pos)
        strand = strands[1]
        ax.margins(0.05)
        ax.scatter(x = modCounts[group][strand], y = scores[group][strand], 
                   s = 0.8, c = scores[group][strand], label = labels[group])
        if pos == 4:
            ax.set_ylabel('predicted \n scores (-)')
            ax.tick_params(left = True, labelleft= True,
                           bottom = True, labelbottom = True)
        elif pos == 5:
            ax.tick_params(left = True, labelleft= False,
                           bottom = True, labelbottom = True)
        else:
            ax.tick_params(left = True, labelleft= False,
                           bottom = True, labelbottom = True)
        ax.tick_params(left = False)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, 30)
        group +=1
    
    group = 0
    for pos in range(7, 10):
        ax = plt.subplot(4,3, pos)
        strand = strands[0]
        ax.margins(0.05)
        ax.scatter(x = modVars[group][strand], y = scores[group][strand], 
                   s = 0.8, c = scores[group][strand], label = labels[group])
        if pos == 7:
            ax.set_ylabel('predicted \n scores (+)')
            ax.tick_params(left = True, labelleft= True,
                           bottom = False, labelbottom = False)
        else:
            ax.tick_params(left = True, labelleft= False,
                           bottom = False, labelbottom = False)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, 1000)
        group +=1
    
    group = 0
    
    for pos in range(10, 13):
        ax = plt.subplot(4,3, pos)
        strand = strands[1]
        ax.margins(0.05)
        ax.scatter(x = modVars[group][strand], y = scores[group][strand], 
                   s = 0.8, c = scores[group][strand], label = labels[group])
        if pos == 10:
            ax.set_ylabel('predicted \n scores (-)')
            ax.tick_params(left = True, labelleft= True,
                           bottom = True, labelbottom = True)
        elif pos == 11:
            ax.set_xlabel('deviation of modifiable positions from center base (AT/TA)')
            ax.tick_params(left = True, labelleft= False,
                           bottom = True, labelbottom = True)
        else:
            ax.tick_params(left = True, labelleft= False,
                           bottom = True, labelbottom = True)
        ax.tick_params(left = False)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, 1000)
        group +=1

    return plt


def computeAUC(scores, true_lables, strands = [-1,1]):
        
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for s in strands:
        y_true = np.array([int(i) for i in true_lables[0][s]] + [int(j) for j in true_lables[1][s]])
        y_test = np.array(scores[0][s] + scores[1][s])
        fpr[s], tpr[s], threshold = roc_curve(y_true, y_test)
        roc_auc[s] = auc(fpr[s], tpr[s])
    
    return fpr, tpr, roc_auc

def plotROC(scores, true_lables):

    fpr, tpr, roc_auc = computeAUC(scores = scores, 
                            true_lables = true_lables)
    axL = plt.subplot(221)
    s = 1
    axL.plot(fpr[s], tpr[s], color="darkorange", lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc[s])
    axL.set_title("ROC of ctrl data (forward)", size = 'medium')

    axR = plt.subplot(222)
    s = -1
    axR.plot(fpr[s], tpr[s], color="darkorange", lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc[s])
    axR.set_title("ROC of ctrl data (reverse)", size = 'medium')

    for ax in [axL, axR]:
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
    plt.tight_layout()