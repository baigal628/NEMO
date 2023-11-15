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

def readGTF(gtfFile, chromPlot, startPlot, endPlot, genePlot, features):

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
                        splitPoint = str(genePlot) + ' "'
                        transcript = line[8].split(';')[3]
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

def plotGtfTrack(plot, gtfFile, region,
                 features = ['CDS', 'start_codon'], adjust_features = [0, 0.25],
                 label_name = True, label_direction = False, colorpalates= ['orange', 'blue'], 
                 thinHeight = 0.2, thickHeight = 0.8, line_width = 0):
    
    
    chrom = region.split(':')[0]
    locus = region.split(':')[1].split('-')
    startPlot, endPlot = int(locus[0]), int(locus[1])
    features, sorted_gtfReads = readGTF(gtfFile, chromPlot = chrom.split('chr')[1], 
                                        genePlot = 'gene_name',
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
                rectangle = mplpatches.Rectangle([blockStart-adjust_features[0], bottom-(thickHeight/2)],
                                                 blockEnd-blockStart+adjust_features[0], thickHeight,
                                    facecolor = colorpalates[0],
                                    edgecolor = 'black',
                                    linewidth = line_width)
                plot.add_patch(rectangle)
        
        # plot feature2
        if features[1] in sorted_gtfReads[transID]:
            Height = 0.5
            blockStarts = np.array(sorted_gtfReads[transID][features[1]][0], dtype = int)
            blockEnds = np.array(sorted_gtfReads[transID][features[1]][1], dtype = int)
            for index in range(0, len(blockStarts), 1):
                blockStart = blockStarts[index]
                blockEnd = blockEnds[index]
                rectangle = mplpatches.Rectangle([blockStart-adjust_features[1], bottom-(thickHeight/2)], 
                                                 blockEnd-blockStart+adjust_features[1], thickHeight,
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

def clusterRead(mtx, readname, outpath, prefix, strand, n_clusters, 
                show_elbow = False, return_cluster = True, print_inertia = False, print_iterations = False):
    
    
    outfile = open(outpath + prefix  +'_clustering' + str(strand) + '.tsv', 'w')
    outfile.write('readID\tcluster\n')
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    print('Imputing missing values...')
    score_mtx = imp.fit_transform(mtx)

    if show_elbow:
        n_reads = len(readname)
        inertias = []

        for i in range(1, n_reads+1):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(score_mtx)
            inertias.append(kmeans.inertia_)

        plt.plot(range(1,n_reads+1), inertias, marker='o')
        plt.title('Elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

    kmeans = KMeans(
        init="random",
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        random_state=42)

    kmeans.fit(score_mtx)
    if print_inertia:
        print('inertia: ', kmeans.inertia_)
    if print_iterations:
        print('iterations: ', kmeans.n_iter_)

    for k,v in zip(readname, kmeans.labels_):
        line = '{}\t{}\t{}\n'.format(k, strand, v)
        outfile.write(line)
    outfile.close()
    
    if return_cluster:
        return kmeans.labels_

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

def collectPred(prediction, bins, step):
    '''
    collectScores function reads prediction as input, filter reads mapped to the given region, modified positions and 
    modified scores into a format that can feed into clustering algorithms.
    '''
    
    with open(prediction, 'r') as predFh:
        readnames, mtxs = defaultdict(list), defaultdict(list) 
        for line in predFh:
            line = line.strip().split('\t')
            readname = line[0]
            strand = line[1]
            binStart = int(line[2])
            if binStart > bins[-1]:
                continue
            
            probs = line[3].split(',')
            binEnd = binStart + step*(len(probs)-1)

            if binEnd < bins[0]:
                continue
            else:
                thisprobs = np.zeros(len(bins))
                i = int((binStart-bins[0])/step)
                if i < 0:
                    probs = probs[-i:]
                    i = 0
                for prob in probs:
                    thisprobs[i]=prob
                    i+=1
                    if i >= len(bins):
                        break
                readnames[strand].append(readname)
                mtxs[strand].append(thisprobs)
        
        for s in mtxs:
            mtxs[s] = np.array(mtxs[s], dtype = float)
    
    return dict(mtxs), dict(readnames)

def plotModTrack(plot, prediction, region, bins, step, outpath, prefix,
            cluster = True, n_clusters = 1, threashold = 0.5,
            annot = '', label_strand = True, label_rname = False,
            colorPalate = {'modT': 'orangered', 'modF':'dodgerblue', 'unMod': 'lightgrey'}, 
            height = 0.8, line_width = 0, ylim = ''):

    chrom = region.split(':')[0]
    locus = region.split(':')[1].split('-')
    pstart, pend = int(locus[0]), int(locus[1])
    
    pbins = bins[int(pstart/step):int(pend/step)+1]
    print('Reading prediction files...')
    mtxs, readnames = collectPred(prediction, pbins, step)
    print('Finished reading prediction files.')
    
    tick_strand, tick_rname, tick_yaxis = [], [],[]
    strandDict = {'-1':'-', '1': '+'}
    bottom = 0
    
    for strand in mtxs:
        mtx, readname = mtxs[strand], readnames[strand]
        
        try:
            assert mtx.shape == (len(readname), len(pbins))
        except:
            print('dimention of score matrix does not match readname and bin length!')

        
        sorted_readIdx = readname

        if cluster:
            print("clustering reads...")
            label = clusterRead(mtx = mtx, readname = readname, outpath = outpath, 
                                 prefix = prefix+'_'+region, strand = strand, n_clusters=n_clusters, show_elbow = False)
            
            sorted_readIdx = [x for _, x in sorted(zip(label, np.arange(0,len(readname))))]
        
        print('plotting modification track on', strandDict[strand], 'strand...')
        
        for readIdx in sorted_readIdx:

            rectangle = mplpatches.Rectangle([pbins[0], bottom-(height/2)], pbins[-1], height,
                                    facecolor = colorPalate['unMod'],
                                    edgecolor = 'grey',
                                    linewidth = line_width)
            
            plot.add_patch(rectangle)

            tick_yaxis.append(bottom)

            if label_strand:
                tick_strand.append(strandDict[strand])
            if label_rname:
                tick_rname.append(readname[readIdx])

            for posIdx in range(len(pbins)):
                if mtx[readIdx,posIdx] >= threashold:
                    color = colorPalate['modT']
                else:
                    color = colorPalate['modF']

                left = pbins[posIdx]
                rectangle = mplpatches.Rectangle([left, bottom-(height/2)], step, height, 
                                                 facecolor = color, edgecolor = 'grey',
                                                 linewidth = line_width)
                plot.add_patch(rectangle)
            bottom +=1
    
    print('finished plotting file: ', prediction)
    
    plot.set_xlim(pstart, pend)
    plot.set_ylabel(annot)
    if not ylim:
        plot.set_ylim(-1,len(tick_yaxis))
    else:
        plot.set_ylim(ylim[0], ylim[1])
    
    plot.tick_params(bottom=False, labelbottom=False,
                   left=True, labelleft=True,
                   right=False, labelright=False,
                   top=False, labeltop=False)
    if label_strand:
        plot.set_yticks(ticks= tick_yaxis, labels = tick_strand)


def plotAllTrack(prediction, gtf, region, bins, step, outpath, prefix, plot_ctrl=False, 
                 figureWidth=5, figureHeight=7, panelWidth=4, panelHeight=1.5):
    if plot_ctrl:
        plt.figure(figsize=(figureWidth,figureHeight))
        panelt = plt.axes([0.5/figureWidth, 6.1/figureHeight, panelWidth/figureWidth, panelHeight/2.5/figureHeight])
        panel0 = plt.axes([0.5/figureWidth, 5.0/figureHeight, panelWidth/figureWidth, panelHeight/1.5/figureHeight])
        panel1 = plt.axes([0.5/figureWidth, 3.4/figureHeight, panelWidth/figureWidth, panelHeight/figureHeight])
        panel2 = plt.axes([0.5/figureWidth, 1.8/figureHeight, panelWidth/figureWidth, panelHeight/figureHeight])
        panel3 = plt.axes([0.5/figureWidth, 0.2/figureHeight, panelWidth/figureWidth, panelHeight/figureHeight])
    else:
        plt.figure(figsize=(figureWidth,figureHeight))
        figureHeight = figureHeight/2
        panel0 = plt.axes([0.5/figureWidth, 3.05/figureHeight, panelWidth/figureWidth, panelHeight/3.5/figureHeight])
        panel1 = plt.axes([0.5/figureWidth, 2.7/figureHeight, panelWidth/figureWidth, panelHeight/4.6/figureHeight])
        panel2 = plt.axes([0.5/figureWidth, 2.35/figureHeight, panelWidth/figureWidth, panelHeight/4.6/figureHeight])
        panel3 = plt.axes([0.5/figureWidth, 0.2/figureHeight, panelWidth/figureWidth, panelHeight*1.4/figureHeight])
        plotGtfTrack(plot = panel0, region = region, gtfFile = gtf)
        plotModTrack(plot=panel3, prediction=prediction, region=region, bins=bins, step=step, outpath=outpath, prefix=prefix)
    return plt


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


