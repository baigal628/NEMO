import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

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

def collectScores(modScores):
    '''
    collectScores reads modScores as input and format reads, modified positions and 
    modified scores into a format that can feed into clustering algorithms.
    '''
    
    with open(modScores, 'r') as msFh:
        readnames, mtxs = defaultdict(list), defaultdict(list) 
        positions = [int(p) for p in msFh.readline().strip().split('\t')[1].split(',')]       
        for line in msFh:
            line = line.strip().split('\t')
            strand = int(line[2])
            readnames[strand].append(line[0])
            mtxs[strand].append(line[5].split(','))
    for s in mtxs:
        mtxs[s] = np.array(mtxs[s], dtype = float)
    
    return dict(mtxs), dict(readnames), positions


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

def plotGtfTrack(plot, gtfFile, chromPlot, startPlot, endPlot,
                 features = ['CDS', 'start_codon'], adjust_features = [0, 0.25],
                 label_name = True, label_direction = False, colorpalates= ['orange', 'blue'], 
                 thinHeight = 0.2, thickHeight = 0.8, line_width = 0):
    
    features, sorted_gtfReads = readGTF(gtfFile, chromPlot = chromPlot.split('chr')[1], 
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

def plotModTrack(plot, startPlot, endPlot, modScores, cluster = True, n_clusters = 3, threashold = 0.6,
                 outpath = '', prefix = '', annot = '', label_strand = True, label_rname = False,
                 colorPalate = {'modT': 'orangered', 'modF':'dodgerblue', 'unMod': 'lightgrey'}, 
                 height = 0.8, width = 1, line_width = 0, ylim = ''):
    
    print('Reading modScore files...')
    mtxs, readnames, positions = collectScores(modScores)
    tick_strand, tick_rname, tick_yaxis = [], [], []
    strandDict = {-1:'-', 1: '+'}
    bottom = 0
    
    for strand in mtxs:
        mtx, readname = mtxs[strand], readnames[strand]
        
        try:
            assert mtx.shape == (len(readname), len(positions))
        except:
            print('dimention of score matrix does not match readname and position length!')

        
        sorted_readIdx = readname

        if cluster:
            print("clustering reads...")
            label = clusterRead(mtx = mtx, readname = readname, outpath = outpath, 
                                 prefix = prefix, strand = strand, n_clusters=n_clusters, show_elbow = False)
            sorted_readIdx = [x for _, x in sorted(zip(label, np.arange(0,len(readname))))]
        print('plotting modification track on', strandDict[strand], 'strand...')
        
        for readIdx in sorted_readIdx:

            rectangle = mplpatches.Rectangle([startPlot, bottom-(height/2)], endPlot, height,
                                    facecolor = colorPalate['unMod'],
                                    edgecolor = 'grey',
                                    linewidth = line_width)
            
            plot.add_patch(rectangle)

            tick_yaxis.append(bottom)

            if label_strand:
                tick_strand.append(strandDict[strand])
            if label_rname:
                tick_rname.append(readname[readIdx])

            for posIdx in range(len(positions)):
                if mtx[readIdx,posIdx] >= threashold:
                    color = colorPalate['modT']
                else:
                    color = colorPalate['modF']

                left = positions[posIdx]+startPlot
                rectangle = mplpatches.Rectangle([left, bottom-(height/2)], width, height, 
                                                 facecolor = color, edgecolor = 'grey',
                                                 linewidth = line_width)
                plot.add_patch(rectangle)
            bottom +=1
    
    print('finished plotting file: ', modScores)
    plot.set_xlim(startPlot, endPlot)
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

def plotModScores(modPredict_pos, modPredict_neg, modPredict_chrom, return_scores = False):
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

def evaluatePrediction(region, sam, sigAlign, label, kmerWindow=80, signalWindow=400, 
                       modBase = ['AT', 'TA'], genome = genome, model = mymodel, weight = myweight):
    
    alignment = getAlignedReads(sam = sam, region = region, genome=genome, print_name=False)
    refSeq = alignment['ref']
    all_scores, modCounts, modVars = defaultdict(list), defaultdict(list), defaultdict(list)
    modPositions = basePos(refSeq, base = modBase)
    count = baseCount(refSeq, base = modBase)
    
    reg = region.split(':')
    chrom, pStart, pEnd = reg[0], int(reg[1].split('-')[0]), int(reg[1].split('-')[1])
    
    for readID, eventStart, sigList, siglenList in parseSigAlign(sigAlign):
        print(readID)
        start_time = time.time()
        print('Start processing ', readID)
        strand = alignment[readID][1]
        
        sigLenList_init = pStart-eventStart-1
        if sigLenList_init > len(siglenList):
            continue
        
        # Position of As, relative to the reference
        modScores = {i:[] for i in modPositions}

        for pos in range(len(refSeq)):
            if pos % 500 == 0:
                print('Predicting at position:', pos)

            # 1. Fetch sequences with kmer window size, this step is optional
            seq = refSeq[pos:pos+kmerWindow]
            
            # 2. Fetch signals with signal window size 
            pos_sigLenList_start = int(sigLenList_init)+pos
            pos_sigLenList_end = pos_sigLenList_start+1

            if pos_sigLenList_start<0: 
                start=0
            else:
                start = int(siglenList[pos_sigLenList_start])
            
            # reached the end of the signal list
            if len(sigList)-start< signalWindow:
                break
            
            end = int(siglenList[pos_sigLenList_end])
            
            # if no signals aligned to this position
            if start == end:
                continue
            
            signals = [float(s) for s in sigList[start:end+signalWindow]]
            
            # 3. Get predicted probability score from machine learning model
            prob = nntPredict(signals, device = device, model = model, weights_path = weight)
            
            # 4. Assign predicted scores to each modPosition
            # modifiable positions [1,3,4,5,7,10,15,16,21,40]
            # kmer position is 2: [2:2+22]
            # modbase_left = 1
            # modbase_right = 9
            # modifiable position within kmer window [3,4,5,7,10,15,16,21]
            modbase_left = bisect.bisect_left(modPositions, pos)
            modbase_right = bisect.bisect_right(modPositions, pos+kmerWindow)
            modbase_count = modbase_right - modbase_left

            #deviation of modifiable position from center point
            mid = int((pos+kmerWindow)/2) # floor
            # total sum of squares
            tss = [np.square(modPos-mid) for modPos in modPositions[modbase_left:modbase_right]]
            variation = np.sum(tss)/(len(tss)-1)

            all_scores[strand].append(prob)
            modVars[strand].append(np.sqrt(variation))
            modCounts[strand].append(modbase_count)
    true_labels = {}
    for s in all_scores:
        true_labels[s] = np.ones(len(all_scores[s]))*label
    
    return all_scores, modVars, modCounts, true_labels


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

def plotROC(scores = [pos_scores, neg_scores], true_lables =  [pos_true_label, neg_true_label]):

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