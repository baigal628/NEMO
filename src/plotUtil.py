import numpy as np

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
    iteration = 0
    with open(modScores, 'r') as msFh:
        reads = {}
        readnames, positions, mtx = [], [], []
        for line in msFh:
            lines = line.split('}')
            for read in lines:
                if read:
                    read = read.split('\t')
                    readnames.append(read[0])
                    modScores = read[4].split('{')[1].split(',')
                    scores = []
                    for i in modScores:
                        pos_score = i.split(':')
                        if iteration == 0:
                            positions.append(int(pos_score[0]))
                        scores.append(float(pos_score[1]))
                    mtx.append(scores)
                    iteration +=1
    return np.array(mtx), readnames, positions


def clusterRead(score_mtx, outpath, prefix, n_clusters=2, show_elbow = False, return_cluster = True):
    
    outfile = open(outpath + prefix + '_clustering.tsv', 'w')
    outfile.write('readID\tcluster\n')
    n_reads = len(readnames)
    
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
    
    for k,v in zip(readnames, kmeans.labels_):
        line = '{}\t{}\n'.format(k,v)
        outfile.write(line)
    outfile.close()
    
    if return_cluster:
        return (kmeans.labels_)