# def exportBedGraph(region, sam, sigAlign, genome, model, weight, kmerWindow=80, signalWindow=400, binSize = 75, modBase = ['AT', 'TA']):
    
#     alignment = getAlignedReads(sam = sam, region = region, genome=genome, print_name=False)
#     refSeq = alignment['ref']
#     all_scores, modCounts, modVars = defaultdict(list), defaultdict(list), defaultdict(list)
#     modPositions = basePos(refSeq, base = modBase)
#     count = baseCount(refSeq, base = modBase)
    
#     reg = region.split(':')
#     chrom, pStart, pEnd = reg[0], int(reg[1].split('-')[0]), int(reg[1].split('-')[1])
    
#     bins = np.arange(pStart, pEnd, binSize)
#     binScores = {bin:0 for bin in bins}
#     binCounts = {bin:0 for bin in bins}

#     for readID, eventStart, sigList, siglenList in parseSigAlign(sigAlign):
#         print(readID)
#         start_time = time.time()
#         print('Start processing ', readID)
#         strand = alignment[readID][1]
        
#         sigLenList_init = pStart-eventStart-1
#         if sigLenList_init > len(siglenList):
#             continue
#         for pos in range(len(refSeq)):
#             if pos % 500 == 0:
#                 print('Predicting at position:', pos)

#             # 1. Fetch sequences with kmer window size, this step is optional
#             seq = refSeq[pos:pos+kmerWindow]
            
#             # 2. Fetch signals with signal window size 
#             signals = fetchSignal(pos, sigLenList_init, siglenList, sigList, signalWindow)
#             if signals == 'del':
#                 continue
#             elif signals == 'end':
#                 break
            
#             # 3. Get predicted probability score from machine learning model
#             prob = nntPredict(signals, device = device, model = model, weights_path = weight)

#             idx = np.searchsorted(bins, pStart+pos, side='right')
#             binScores[bins[idx-1]] +=prob
#             binCounts[bins[idx-1]] +=1                    
#     return binScores, binCounts

# def writeBedGraph(bedGraphHeader, binScores, binCounts, binSize, chrom, outfile):
#     outFh = open(outfile, 'w')
#     for k,v in bedGraphHeader.items():
#         if v:
#             line = k + '=' + v + ' '
#             outFh.write(line)
#     outFh.write('\n')
#     for chrStart in binScores.keys():
#         chrEnd = chrStart + binSize
#         score = "%.3f" % (binScores[chrStart]/binCounts[chrStart])
#         line = '{chr}\t{start}\t{end}\t{score}\n'.format(chr = chrom, start = chrStart,  end = chrEnd, score = score)
#         outFh.write(line)
#     outFh.close()

# bedGraphHeader = {'track type':'bedGraph', 
#                   'name':'chrom_bin75', 
#                   'description':'addseq',
#                   'visibility':'', 
#                   'color':'r', 
#                   'altColor':'r', 
#                   'priority':'', 
#                   'autoScale':'off', 
#                   'alwaysZero':'off', 
#                   'gridDefault':'off', 
#                   'maxHeightPixels':'default', 
#                   'graphType':'bar',
#                   'viewLimits':'upper',
#                   'yLineMark':'',
#                   'yLineOnOff':'on',
#                   'windowingFunction':'mean',
#                   'smoothingWindow':'on'
#                  }