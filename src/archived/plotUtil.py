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

def predToMtx(infile, pregion, bins, outpath = '', prefix = '', step = 40, impute = True, mystrand = '',
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
    all_scores = []
    with open(infile, 'r') as predFh:
        for line in predFh:
            bin_scores = np.empty(len(bins), dtype = float) * np.nan
            line = line.strip().split('\t')
            readname = line[0]
            strand = line[1]
            if mystrand:
                if strand != mystrand:
                    continue
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
                    prob = float(prob)
                    if not np.isnan(prob):
                        all_scores.append(float(prob))
                    bin_scores[i] = prob
                    i +=1
                    if i >= len(bins):
                        break
            readnames.append(readname)
            strands.append(strand)
            mtx.append(bin_scores)
    
    kde = stats.gaussian_kde(all_scores)
    # Create a range of values for x-axis
    x = np.linspace(-0.01,1.01, 100)
    plt.plot(x, kde(x))
    
    mtx = np.array(mtx, dtype = float)  
    mtx = mtx[:,rstart:rend]
    bins = bins[rstart:rend]
    readnames = np.array(readnames, dtype = int)
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

def clusterRead(predict, outpath, prefix, pregion, bins, step = 40, n_cluster = '', random_state = 42, method = '', show_elbow = True, nPC= 5, na_thred = 0.5, mystrand = ''):
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
        na_thred: percent of missing bins allowed each read. The less this value is, the more stricter filtering is.
    output:
        outpath: output file path
        prefix: output file prefix
    return:
        readnames, strands, mtx, bins
    '''
    
    print('preprocessing input matrix...')
    if method == 'pca':
        print('Reading prediction file and outputing matrix...')
        prefix = prefix + "_method_pca"
        readnames, strands, mtx, bins = predToMtx(infile=predict, pregion=pregion, bins=bins, outpath=outpath, prefix=prefix, step=step, na_thred=na_thred, mystrand=mystrand)
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
        plt.savefig(outpath + prefix + pregion + "_pca.pdf")
        plt.close()
        
    elif method == 'cor':
        print('Reading prediction file and outputing matrix...')
        prefix = prefix + "_method_cor"
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
            symbol = '+' if strands[i] == 1 else '-'
            label_yaxis.append(symbol)
        elif label == 'readname':
            tick_yaxis.append(bottom)
            label_yaxis.append(str(readnames[i]))
        
        if labels[i] != thiscluster:

            thiscluster = labels[i]
            label_clusters.append('c'+str(labels[i]))
            
            if thiscluster:
                aggregate = count/total
                if np.max(total) < 3:
                    aggregate = np.zeros(len(bins))
                # aggregate = ((aggregate-np.min(aggregate))/(np.max(aggregate)-np.min(aggregate)))
                for pos in range(len(bins)):
                    rectangle = mplpatches.Rectangle([left, bottom-height*0.5], step, aggregate[pos]*extend,
                                                     facecolor = 'silver', edgecolor = 'black', linewidth = 0.5)
                    ax.add_patch(rectangle)
                    left += step
                left = bins[0]
                total, count = np.zeros(len(bins), dtype = float), np.zeros(len(bins), dtype = int)
                bottom +=(np.max(aggregate)*extend+1)
            
            tick_clusters.append(bottom)
        
        for pos in range(len(bins)):
            score = mtx[i, pos]
            if score >= colorRange[1]:
                count[pos] += 1
            total[pos] += 1
            if np.isnan(score):
                col = 'white'
            else:
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
    
    aggregate = count/total
    if np.max(total) < 3:
        aggregate = np.zeros(len(bins))
    left = bins[0]
    for pos in range(len(bins)):
        rectangle = mplpatches.Rectangle([left, bottom-height*0.5], step, aggregate[pos]*extend, 
                                         facecolor = 'silver', edgecolor = 'black', linewidth = 0.5)
        ax.add_patch(rectangle)
        left += step
    bottom +=(np.max(aggregate)*extend+1)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(-1.5, bottom)

    ax.tick_params(
        bottom=True, labelbottom=True,
        left=False, labelleft=True,
        right=False, labelright=False,
        top=False, labeltop=False)
    
    ax.set_yticks(ticks= tick_clusters, labels = label_clusters)
    if label in ['readname', 'strand']:
            ax.set_yticks(ticks= tick_yaxis, labels = label_yaxis)
    ax.set_xticks(ticks= np.arange(bins[0], bins[-1], xticks_space))
    ax.set_xticklabels(ax.get_xticks(), rotation = 50)

def plotbdgTrack(plot, bdg, region, step = 1, scale = 1000, header = False, col = 'grey', annot = '', ylim = ''):
    
    chrom = region.split(':')[0]
    locus = region.split(':')[1].split('-')
    pstart, pend = int(locus[0]), int(locus[1])
    ymax = 0
    
    print('plotting ' , bdg,  '...')
    with open(bdg, 'r') as bdgFh:
        if header:
            header = bdgFh.readlines(1)
        for line in bdgFh:
            line = line.strip().split('\t')
            if line[0] != chrom:
                continue
            start = int(line[1])
            end =  int(line[2])
            if end < pstart:
                continue
            elif start > pend:
                break
            else:
                prob = float(line[3])
                height = min(1.0, prob/scale)
                if height > ymax:
                    ymax = height
                left = max(start, pstart)
                rectangle = mplpatches.Rectangle([left, 0], end-left, height,
                        facecolor = col,
                        edgecolor = 'grey',
                        linewidth = 0)
                plot.add_patch(rectangle)
    
    plot.set_xlim(pstart, pend)
    if ylim:
        plot.set_ylim(0,ylim+0.1)
    plot.set_ylim(0,ymax+0.1)
    plot.tick_params(bottom=False, labelbottom=False,
                   left=False, labelleft=False,
                   right=False, labelright=False,
                   top=False, labeltop=False)
    plot.set_ylabel(annot)
    print('Finished plotting ' , bdg,  '!')

def plotlegend(ax, colorRange, colorPalette):
    y_ticks_axis, Y_ticks_labels = [0], ['0']
    bottom=0
    height = 1
    i = colorRange[0]
    plotcolor = True
    (R,G,B) = colorMap(palette = colorPalette)

    while plotcolor:
        if i == colorRange[1]:
            y_ticks_axis.append(bottom)
            Y_ticks_labels.append(str(i))
            color = 80
            height = bottom
        elif i == colorRange[2]:
            y_ticks_axis.append(bottom)
            Y_ticks_labels.append(str(i))
            color = 100
        else:
            color = int(i*100)
        # print(i, color)
        col = (R[color],G[color],B[color])
        rectangle = mplpatches.Rectangle([0, bottom], 1, height,
                                     facecolor = col, edgecolor = 'silver',
                                     linewidth = 0)
        ax.add_patch(rectangle)
        bottom +=height
        if i == colorRange[2]:
            plotcolor = False
        elif i == colorRange[1]:
            i = colorRange[2]
        else:
            i+=0.01
            i = round(i, 2)
    
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0,bottom)
    
    ax.tick_params(
        bottom=False, labelbottom=False,
        left=False, labelleft=False,
        right=True, labelright=True,
        top=False, labeltop=False)
    
    ax.set_yticks(ticks = y_ticks_axis, labels = Y_ticks_labels)

def plotAllTrack(prediction, gtfFile, bins, pregion, na_thred = 0.5, 
                 step = 20, outpath = '', prefix = '', ncluster = 3, method = '', 
                 subset = '', colorPalette = 'viridis', colorRange = (0.3, 0.5, 0.6), vlines = '',
                 gtfFeatures = ['CDS', 'start_codon'],  genePlot = {'CDS': 'gene_name', 'start_codon': 'gene_name'}, 
                 geneSlot = {'CDS': 3, 'start_codon': 3}, gtfHeight = [0.8, 0.8], adjust_features = '',
                 trackHeight = 0.5, fontsize=10, track_ylim_adjust = 0.5, track_agg_adjust = 0.4, xticks_space = 120,
                 fig_size = '', savefig = True, outfig = '', dpi = 800, seed = 42, modtrack_label = ''):

    print('Start clustering reads...')
    labels, readnames, strands, mtx, bins = clusterRead(predict=prediction, outpath=outpath, prefix=prefix, pregion=pregion, bins=bins, step =step, na_thred=na_thred,
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
    ax1 = plt.axes((0.1, 0.2 , 0.73, 0.6), frameon=False)
    ax2 = plt.axes((0.1, 0.8, 0.73, 0.16), frameon=False)
    ax3 = plt.axes((0.85, 0.4, 0.05, 0.16), frameon=True)

    plotModTrack(ax=ax1, labels=labels, readnames=readnames, strands=strands, mtx=mtx, bins=bins, step=step,
                 ylim_adjust=track_ylim_adjust, agg_adjust=track_agg_adjust, label=modtrack_label,
                 colorRange = colorRange, colorPalette=colorPalette, height=trackHeight, xticks_space=xticks_space)
    
    plotGtfTrack(ax2, gtfFile, pregion, Height = gtfHeight, features = gtfFeatures, 
                 genePlot = genePlot, geneSlot = geneSlot, adjust_features=adjust_features)
    
    plotlegend(ax3, colorRange, colorPalette)

    if vlines:
        for label, vl in vlines.items():
            ax2.axvline(x = vl, color = 'black', linestyle = 'dashed')
            ax2.text(vl+12, 0.7, label, fontsize=fontsize)
    if savefig:
        if not outfig:
            outfig = outpath + prefix + '_' + pregion + '_' + method + '_clustered_reads.pdf'
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

def plotAggregateModBam(modbam, bed, window, sw=10, step =20, end = False, space=147, thred = 0.45, 
                  labels = ('distance to tss (bp)', 'prediction score'), 
                  col = {'chrom':0, 'start':1, 'end':2, 'strand':4}, outpath='', prefix ='', chrom = ''):

    halfwindow = int(window/2)
    hsw = int(round(sw/2))
    tsspos = {}

    compbase = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N'}
    def getcomp(seq):
        newseq = []
        for base in seq: newseq.append(compbase[base])
        return ''.join(newseq)#newseq[::-1]
    
    
    
    with open(bed, 'r') as infile:
        for line in infile:
            if 'track' not in line:
                line = line.strip().split()
                chr, dir = line[col['chrom']], line[col['strand']]
                if end: pos = int(line[col['start']]) if dir == '-' else int(line[col['end']])
                else: pos = int(line[col['start']]) if dir == '+' else int(line[col['end']])
                if chr not in tsspos:
                    tsspos[chr] = []
                tsspos[chr].append((pos - halfwindow, pos+halfwindow, dir))
    
    tssscores = []
    for i in range(window+1):
        tssscores.append([])
    
    typesOfMods = {'5mC':[('C', 0, 'm')], '5hmC': [('C', 0, 'h')], '5fC': [('C', 0, 'f')], '5caC': [('C', 0, 'c')],
                   '5hmU': [('T', 0, 'g')], '5fU': [('T', 0, 'e')], '5caU': [('T', 0, 'b')],
                   '6mA': [('A', 0, 'a'), ('A', 0, 'Y')], '8oxoG': [('G', 0, 'o')], 'Xao': [('N', 0, 'n')]}

    tssscores = []
    for i in range(window+1):
        tssscores.append([])
    samfile = pysam.AlignmentFile(modbam, "rb")
    for s in samfile:
        chr = s.reference_name
        if not s.is_secondary and chr in tsspos:
            alignstart, alignend = s.reference_start, s.reference_end
            hassite = False
            for pos in tsspos[chr]:
                if alignstart < pos[0] and pos[1] < alignend: hassite = True
            if hassite:
                readname = s.query_name
                cigar = s.cigartuples
                posstag = typesOfMods['6mA']
                if s.is_reverse: posstag = [(x[0], 1, x[2]) for x in posstag]
                ml = None
                for t in posstag:
                    if t in s.modified_bases:
                        ml = s.modified_bases[t]
                        break
                if not ml:
                    print(readname, 'does not have modification information', s.modified_bases.keys())
                    continue

                if s.has_tag('MM'):
                    skippedBase = -1 if s.get_tag('MM').split(',', 2)[0][-1] == '?' else 0
                elif s.has_tag('Mm'):
                    skippedBase = -1 if s.get_tag('Mm').split(',', 2)[0][-1] == '?' else 0
                else:
                    continue

                seq = s.query_sequence
                seqlen = len(seq)
                if s.is_reverse:  ###need to get compliment of sequence, but not reverse!!
                    seq = getcomp(seq)

                seqApos = []
                c = 0
                for b in seq:
                    if b == 'A':
                        seqApos.append(c)
                    c += 1

                ml = dict(ml)
                for i in seqApos:
                    if i not in ml:
                        ml[i] = skippedBase

                ref, quer = 0, 0
                posOnGenome = []
                for block in cigar:
                    if block[0] in {0, 7, 8}:  # match, consumes both
                        for i in range(block[1]):
                            if quer in ml: posOnGenome.append([ref + alignstart, ml[quer]])
                            ref += 1
                            quer += 1
                    elif block[0] in {1, 4}:  # consumes query
                        quer += block[1]
                    elif block[0] in {2, 3}:  # consumes reference
                        ref += block[1]
                dirtowrite = '-' if s.is_reverse else '+'
                posdict = dict(posOnGenome)
                
                for coord in tsspos[chr]:
                    start, stop, dir = coord[0], coord[1], coord[2]
                    if start > alignstart and stop < alignend:
                        for pos in range(start, stop+1):
                            if pos in posdict:
                                thisscore = posdict[pos]
                                if dir == '+':
                                    tssscorepos = pos-start
                                else:
                                    tssscorepos = window - (pos-start)
                                tssscores[tssscorepos].append(thisscore)

    samfile.close()
    
    
    xval, yval = [], []

    hsw = int(round(sw/2))
    tssscores = [np.mean(x) if len(x) > 0 else 0 for x in tssscores]
    
    for i in range(hsw, (window+1)-(hsw+1), int(round(hsw/2))):
        thesescores = tssscores[i-hsw:i+hsw]
        avg = sum(thesescores)/len(thesescores)
        yval.append(avg)
        xval.append(i-halfwindow)
    
    plt.figure(figsize=(6,4))
    plt.plot(xval,yval)
    plt.xticks(np.concatenate((np.flip(np.arange(0, -halfwindow, -space)[1:]), np.arange(0, halfwindow, space)), axis=0), rotation='vertical')
    plt.grid(alpha=0.5,axis = 'x')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(outpath+prefix+'_aggregate.pdf', dpi = 200)
    plt.close()
    
    return (xval, yval)

def plotAggregate_nuc(pred, bed, window, sw=10, step =20, end = False, space=147,
                      labels = ('distance to tss (bp)', 'prediction score'), 
                      col = {'chrom':0, 'start':1, 'end':2, 'strand':4}, outpath='', prefix ='', chrom = ''):

    
    halfwindow = int(window/2)
    hsw = int(round(sw/2))
    tsspos = {}
    
    with open(bed, 'r') as infile:
        for line in infile:
            if 'track' not in line:
                line = line.strip().split()
                chr, dir = line[col['chrom']], line[col['strand']]
                if end: pos = int(line[col['end']]) if dir == '-' else int(line[col['start']])
                else: pos = int(line[col['start']]) if dir == '+' else int(line[col['end']])
                if chr not in tsspos:
                    tsspos[chr] = []
                tsspos[chr].append((pos - halfwindow, pos+halfwindow, dir))
    
    tssscores = []
    for i in range(window+1):
        tssscores.append([])

    posOnGenome = []
    with open(pred) as infile:
        for line in infile:
            line=line.strip().split('\t')
            if chrom:
                chr = chrom
            else:
                chr = line[0]
            astart = int(line[1])
            aend = int(line[2])
            prob = float(line[3])
            posOnGenome.append([astart, prob])
    posdict = dict(posOnGenome)
            
    for coord in tsspos[chr]:
        start, stop, dir = coord[0], coord[1], coord[2]
        for pos in range(start, stop+1):
            if pos in posdict:
                thisscore = posdict[pos]
                if dir == '+':
                    tssscorepos = pos-start
                else:
                    tssscorepos = window - (pos-start)
                tssscores[tssscorepos].append(thisscore)
        
    tsscores = [sum(x)/len(x) if len(x) > 0 else 0 for x in tssscores]
    xval, yval = [], []
    
    for i in range(hsw, (window+1)-(hsw+1), int(round(hsw/2))):
        thesescores = tsscores[i-hsw:i+hsw]
        avg = sum(thesescores)/len(thesescores)
        yval.append(avg)
        xval.append(i-halfwindow)
    
    plt.figure(figsize=(6,4))
    plt.plot(xval,yval)
    plt.xticks(np.concatenate((np.flip(np.arange(0, -halfwindow-1, -space)[1:]), np.arange(0, halfwindow+1, space)), axis=0), rotation='vertical')
    plt.grid(alpha=0.5,axis = 'x')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(outpath+prefix+'_aggregate.pdf', dpi = 200)
    plt.close()
    
    return (xval, yval)

def plotAggregate(pred, bed, window, sw=10, step =20, end = False, space = 147, thred=0.45, labels=('distance to tss (bp)', 'prediction score'),
                  col = {'chrom':0, 'start':1, 'end':2, 'strand':4}, outpath='', prefix ='', chrom = ''):
    
    halfwindow = int(window/2)
    hsw = int(round(sw/2))
    tsspos = {}
    
    with open(bed, 'r') as infile:
        for line in infile:
            if 'track' not in line:
                line = line.strip().split()
                chr, dir = line[col['chrom']], line[col['strand']]
                if end: pos = int(line[col['end']]) if dir == '-' else int(line[col['start']])
                else: pos = int(line[col['start']]) if dir == '+' else int(line[col['end']])
                if chr not in tsspos:
                    tsspos[chr] = []
                tsspos[chr].append((pos - halfwindow, pos+halfwindow, dir))
    
    tssscores = []
    for i in range(window+1):
        tssscores.append([])

    with open(pred) as infile:
        for line in infile:
            posOnGenome = []
            line=line.strip().split('\t')
            if chrom:
                chr = chrom
            else:
                chr = line[4]
            astart = int(line[2])
            probs = line[3].split(',')
            aend = astart + step*(len(probs)-1)
            for i in range(len(probs)):
                prob = float(probs[i])
                if np.isnan(prob):
                    continue
                if thred:
                    prob = 1 if prob >= thred else 0
                start = astart + i*step
                for j in range(step):
                    posOnGenome.append([start+j, prob])
            posdict = dict(posOnGenome)
            
            for coord in tsspos[chr]:
                start, stop, dir = coord[0], coord[1], coord[2]
                if start > astart and stop < aend:
                    for pos in range(start, stop+1):
                        if pos in posdict:
                            thisscore = posdict[pos]
                            if dir == '+':
                                tssscorepos = pos-start
                            else:
                                tssscorepos = window - (pos-start)
                            tssscores[tssscorepos].append(thisscore)
        
        tsscores = [sum(x)/len(x) if len(x) > 0 else 0 for x in tssscores]
        xval, yval = [], []
        
    for i in range(hsw, (window+1)-(hsw+1), int(round(hsw/2))):
        thesescores = tsscores[i-hsw:i+hsw]
        avg = sum(thesescores)/len(thesescores)
        yval.append(avg)
        xval.append(i-halfwindow)
    
    plt.figure(figsize=(6,4))
    plt.plot(xval,yval)
    plt.xticks(np.concatenate((np.flip(np.arange(0, -halfwindow-1, -space)[1:]), np.arange(0, halfwindow+1, space)), axis=0), rotation='vertical')
    plt.grid(alpha=0.5,axis = 'x')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(outpath+prefix+'_aggregate.pdf', dpi = 200)
    plt.close()
    
    return (xval, yval)


def plotmetagene(predout, bed, genome, window, method, sw='', space=150, labels=('distance to +1 nuc (bp)', 'prediction score'), 
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