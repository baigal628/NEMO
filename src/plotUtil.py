import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from collections import defaultdict 

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
            plot.text(x = start-1, y = bottom, s = transID, ha = 'right', va = 'center', size = 'small')
            
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