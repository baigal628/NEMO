# def evaluatePrediction(region, sam, sigAlign, label, genome, model, weight, kmerWindow=80, signalWindow=400, 
#                        modBase = ['AT', 'TA']):
    
#     alignment = getAlignedReads(sam = sam, region = region, genome=genome, print_name=False)
#     refSeq = alignment['ref']
#     all_scores, modCounts, modVars = defaultdict(list), defaultdict(list), defaultdict(list)
#     modPositions = basePos(refSeq, base = modBase)
#     count = baseCount(refSeq, base = modBase)
    
#     reg = region.split(':')
#     chrom, pStart, pEnd = reg[0], int(reg[1].split('-')[0]), int(reg[1].split('-')[1])
    
#     for readID, eventStart, sigList, siglenList in parseSigAlign(sigAlign):
#         print(readID)
#         start_time = time.time()
#         print('Start processing ', readID)
#         strand = alignment[readID][1]
        
#         sigLenList_init = pStart-eventStart-1
#         if sigLenList_init > len(siglenList):
#             continue
        
#         # Position of As, relative to the reference
#         modScores = {i:[] for i in modPositions}

#         for pos in range(len(refSeq)):
#             if pos % 500 == 0:
#                 print('Predicting at position:', pos)

#             # 1. Fetch sequences with kmer window size, this step is optional
#             seq = refSeq[pos:pos+kmerWindow]
            
#             # 2. Fetch signals with signal window size 
#             pos_sigLenList_start = int(sigLenList_init)+pos
#             pos_sigLenList_end = pos_sigLenList_start+1

#             if pos_sigLenList_start<0: 
#                 start=0
#             else:
#                 start = int(siglenList[pos_sigLenList_start])
            
#             # reached the end of the signal list
#             if len(sigList)-start< signalWindow:
#                 break
            
#             end = int(siglenList[pos_sigLenList_end])
            
#             # if no signals aligned to this position
#             if start == end:
#                 continue
            
#             signals = [float(s) for s in sigList[start:end+signalWindow]]
            
#             # 3. Get predicted probability score from machine learning model
#             prob = nntPredict(signals, device = device, model = model, weights_path = weight)
            
#             # 4. Assign predicted scores to each modPosition
#             # modifiable positions [1,3,4,5,7,10,15,16,21,40]
#             # kmer position is 2: [2:2+22]
#             # modbase_left = 1
#             # modbase_right = 9
#             # modifiable position within kmer window [3,4,5,7,10,15,16,21]
#             modbase_left = bisect.bisect_left(modPositions, pos)
#             modbase_right = bisect.bisect_right(modPositions, pos+kmerWindow)
#             modbase_count = modbase_right - modbase_left

#             #deviation of modifiable position from center point
#             mid = int((pos+kmerWindow)/2) # floor
#             # total sum of squares
#             tss = [np.square(modPos-mid) for modPos in modPositions[modbase_left:modbase_right]]
#             variation = np.sum(tss)/(len(tss)-1)

#             all_scores[strand].append(prob)
#             modVars[strand].append(np.sqrt(variation))
#             modCounts[strand].append(modbase_count)
#     true_labels = {}
#     for s in all_scores:
#         true_labels[s] = np.ones(len(all_scores[s]))*label
    
#     return all_scores, modVars, modCounts, true_labels