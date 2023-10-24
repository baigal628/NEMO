import numpy as np
from sklearn.cluster import KMeans

def readGTF(gtfFile, chromPlot, startPlot, endPlot, features = ['exon', 'CDS']):

    gtfReads = {}
    gene = ''
    count = 0

    with open(gtfFile) as gtfFh:
        for line in gtfFh:
            if '##' in line:
                pass
            else:
                line = line.split('\t')
                chrom = str(line[0])
                start = int(line[3])
                end = int(line[4])
                feature = line[2]
                if chrom != chromPlot:
                    pass
                else:
                    if feature in features:
                        geneID = line[8].split(';')[0].split('gene_id "')[1].split('"')[0]
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
        readnames, mtxs = {-1:[], 1:[]}, {-1:[], 1:[]}
        positions = [int(p) for p in msFh.readline().strip().split('\t')[1].split(',')]       
        for line in msFh:
            line = line.strip().split('\t')
            strand = int(line[2])
            readnames[strand].append(line[0])
            mtxs[strand].append(line[5].split(','))
    for s in (-1,1):
        mtxs[s] = np.array(mtxs[s], dtype = float)
    
    return mtxs, readnames, positions


def clusterRead(mtxs, readnames, outpath, prefix, n_clusters=2, show_elbow = False, return_cluster = True):
    
    outfile = open(outpath + prefix + '_clustering.tsv', 'w')
    outfile.write('readID\tcluster\n')
    labels={}
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    for strand in readnames:
        readname = readnames[strand]
        n_reads = len(readname)
        
        print('Imputing missing values...')
        score_mtx = imp.fit_transform((mtxs[strand]))

        if show_elbow:
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
        print('inertia: ', kmeans.inertia_)
        print('iterations: ', kmeans.n_iter_)
        
        classification = kmeans.labels_
        for k,v in zip(readname, classification):
            line = '{}\t{}\t{}\n'.format(k, strand, v)
            outfile.write(line)
        labels[strand] = classification
    
    outfile.close()
    if return_cluster:
        return labels
    

def plotGtfTrack(plot, gtfFile, chromPlot, startPlot, endPlot):
    
    features, sorted_gtfReads = readGTF(gtfFile, chromPlot = chromPlot, 
                                        startPlot = startPlot, endPlot = endPlot)
    
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
            elif start >= yRightMost[y]:
                bottom = y
                yRightMost[y] = end
                break
            else:
                y +=1
        thinheight = 0.05
        line_width = 0
        rectangle = mplpatches.Rectangle([start, bottom-(thinheight/2)], end - start, thinheight,
                                        facecolor = 'grey',
                                        edgecolor = 'black',
                                        linewidth = line_width)
        plot.add_patch(rectangle)
        if features[0] in sorted_gtfReads[transID]:
            Height = 0.25
            blockStarts = np.array(sorted_gtfReads[transID]['exon'][0], dtype  = int)
            blockEnds = np.array(sorted_gtfReads[transID]['exon'][1], dtype = int)
            for index in range(len(blockStarts)):
                blockStart = blockStarts[index]
                blockEnd = blockEnds[index]
                rectangle = mplpatches.Rectangle([blockStart, bottom-(Height/2)], blockEnd-blockStart, Height,
                                    facecolor = 'blue',
                                    edgecolor = 'black',
                                    linewidth = line_width)
                plot.add_patch(rectangle)
        if features[1] in sorted_gtfReads[transID]:
            Height = 0.5
            blockStarts = np.array(sorted_gtfReads[transID]['CDS'][0], dtype = int)
            blockEnds = np.array(sorted_gtfReads[transID]['CDS'][1], dtype = int)
            for index in range(0, len(blockStarts), 1):
                blockStart = blockStarts[index]
                blockEnd = blockEnds[index]
                rectangle = mplpatches.Rectangle([blockStart, bottom-(Height/2)], blockEnd-blockStart, Height,
                                    facecolor = 'orange',
                                    edgecolor = 'black',
                                    linewidth = line_width)
                plot.add_patch(rectangle)

    plot.set_xlim(startPlot, endPlot)
    plot.set_ylim(-1,3)

def plotModTrack(plot, startPlot, endPlot, modScores, cluster = True, n_clusters = 3, 
                 outpath = '',
                 prefix = '',
                 col = {'modT': 'orangered', 'modF':'dodgerblue', 'unMod': 'lightgrey'}, 
                 height = 0.8, width = 1, line_width = 0, threashold = 0.6):
    
    print('Reading modScore files...')
    mtxs, readnames, positions = collectScores(modScores)
    sorted_readIdx_for ,sorted_readIdx_rev = readnames[1], readnames[-1]
    
    try:
        assert score_mtx.shape == (len(readnames), len(positions))
    except:
        print('dimention of score matrix does not match readnames and positions length!')
    
    if cluster:
        print("clustering reads...")
        labels = clusterRead(mtxs = mtxs,  readnames = readnames, outpath = outpath, 
                             prefix = prefix, n_clusters=3, show_elbow = False)
        
        sorted_readIdx_for = [x for _, x in sorted(zip(labels[1],np.arange(0,len(readnames[1]))))]
        sorted_readIdx_rev = [x for _, x in sorted(zip(labels[-1],np.arange(0,len(readnames[-1]))))]
    
    print('plotting modification track')
    colorPalate = col
    bottom = 0
    
    print('plotting forward')
    for readIdx in sorted_readIdx_for:
    
        rectangle = mplpatches.Rectangle([startPlot, bottom-(height/2)], endPlot, height,
                                facecolor = colorPalate['unMod'],
                                edgecolor = 'grey',
                                linewidth = line_width)
        plot.add_patch(rectangle)
        for posIdx in range(len(positions)):
            if mtxs[1][readIdx,posIdx] >= threashold:
                color = colorPalate['modT']
            else:
                color = colorPalate['modF']

            left = positions[posIdx]+startPlot
            rectangle = mplpatches.Rectangle([left, bottom-(height/2)], width, height, 
                                             facecolor = color, edgecolor = 'grey', linewidth = line_width)
            plot.add_patch(rectangle)
            plot.text(x = startPlot-1, y = bottom, s = '+', ha = 'right')
            plot.text(x = startPlot-2, y = bottom, s = readnames[1][readIdx], ha = 'right')
        bottom +=1
    
    print('plotting reverse')
    for readIdx in sorted_readIdx_rev:
    
        rectangle = mplpatches.Rectangle([startPlot, bottom-(height/2)], endPlot, height,
                                facecolor = colorPalate['unMod'],
                                edgecolor = 'grey',
                                linewidth = line_width)
        plot.add_patch(rectangle)
        
        for posIdx in range(len(positions)):
            if mtxs[-1][readIdx,posIdx] >= threashold:
                color = colorPalate['modT']
            else:
                color = colorPalate['modF']

            left = positions[posIdx]+startPlot
            rectangle = mplpatches.Rectangle([left, bottom-(height/2)], width, height, 
                                             facecolor = color, edgecolor = 'grey', linewidth = line_width)
            plot.add_patch(rectangle)
            plot.text(x = startPlot-1, y = bottom, s = '-', ha = 'right')
            plot.text(x = startPlot-2, y = bottom, s = readnames[-1][readIdx], ha = 'right')
        bottom +=1

    plot.set_xlim(startPlot, endPlot)
    plot.set_ylim(-1,len(readnames))