def parseEventAlign(eventAlign, outfile, reads, print_sequence = False, header = True):
    '''
    This function reads nanopolish eventalign file, aggregates signals and the number of 
    signals correspinding to one base movement for read in readname list.
    
    input:
        eventAlign: nanopolish eventalign output file.
    optional:
        print_sequence: if True, kmer sequence will be included in outfile.
    output: 
        outfile: siganlAlign.tsv with format: readname\tchrom\teventStart(reference)\tsigList\tsigLenList

    E.g.    read1  ACGTGGCTGA
            events ACGTG
                    CGTGG
                     GTGGC
                      TGGCT
                       GGCTG
                        GCTGA
            sigLen  23
                     45
                      61
                       78
                        101
    '''
    
    if outfile:
        outf = open(outfile, 'w')
    
    read = ''
    sequence = ''
    c = 0
    
    with open(eventAlign, 'r') as inFile:
        if header:
            header = inFile.readline()
        for line in inFile:
            line = line.strip().split('\t')
            thisread = line[3]
            thischrom = line[0]
            c+=1
            if c%10000000 == 0:
                print(c/1000000, ' M lines have passed.')
            
            if thisread not in reads:
                continue
            if thisread != read:

                # parsed read exist
                if sequence:
                    # Set variables back to initial state
                    if print_sequence:
                        out = "{}\t{}\t{}\t{}\t{}\t{}\n".format(reads[read], chrom, eventStart, sequence, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
                    else:
                        out = "{}\t{}\t{}\t{}\t{}\n".format(reads[read], chrom, eventStart, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
                    if outfile:
                        outf.write(out)
                    read = ''
                    sequence = ''
                    sigList = []
                    sigLenList = []

                # very first read
                read = thisread
                chrom = thischrom
                eventStart = line[1]
                start = line[1]
                kmer = line[2]

                # signals are stored in column 13/15 and are separated my comma
                sigList = [float(i) for i in line[-1].split(',')]
                sigLen = len(sigList)
                sigLenList = [sigLen]
                sequence = kmer
            
            # next kmer within the same read
            else:
                signals = [float(i) for i in line[-1].split(',')]
                # or signalList += signals
                sigList.extend(signals)
                # signalLength records the number of signals for one base movement
                sigLen += len(signals)

                # If different kmer
                if (line[1], line[2]) != (start, kmer):
                    deletion = int(line[1]) - int(start) - 1
                    # id there is a deletion in eventalign file
                    if deletion > 0:
                        sequence += deletion*'D'
                        for i in range(deletion):
                            sigLenList.append(sigLenList[-1])
                    start = line[1]
                    kmer = line[2]
                    sequence += kmer[-1]
                    sigLenList.append(sigLen)
                # If same kmer
                else:
                    # Update the number of signals matched to previous kmer
                    sigLenList[-1]=sigLen
        if sequence:
            if print_sequence:
                out = "{}\t{}\t{}\t{}\t{}\t{}\n".format(reads[read], chrom, eventStart, sequence, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
            else:
                out = "{}\t{}\t{}\t{}\t{}\n".format(reads[read], chrom, eventStart, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
            if outfile:
                outf.write(out)
    outf.close()


def parseSigAlign(sigalign, alignment):
    '''
    This function is a iterator that reads _sig.tsv file, and output readID, aStart, aEnd, strand, sigList, siglenList.
    
    Input:
        sigalign: sig.tsv file
        alignment: a dictionary storing aligned reads and their alignment information.
    Output:
        aStart: start of alignment in reference genome
        aEnd: end of alignment in reference genome
        strand: strand
        sigList = [00,01,02,03,04,05,..]
        sigLenList = [12,32,51,71,96,26,136]
        GTCGA -> sigLen[51:51+400], sigLen[71:71+400] -> sigLenList[3-1]
        TCGAA -> sigLen[71:71+400], sigLen[96:96+400]
    '''

    with open(sigalign) as sigalignFh:
        for line in sigalignFh:
            line = line.strip().split('\t')
            readID = int(line[0])
            # eventStart = int(line[2])
            if readID not in alignment:
                continue
            sigList = line[3].split(',')
            siglenList = line[4].split(',')
            aStart = int(alignment[readID][1])
            aEnd = int(alignment[readID][2])
            strand = alignment[readID][3]

            yield (readID, aStart, aEnd, strand, sigList, siglenList)


def fetchSignal(start, end, sigLenList, sigList, signalWindow):
    '''
    fetchSignal return a list of signals that are aligned to the givnen position.
    Input:
        start: start position to fetch signals
        sigLenList: a list with length of signals aligned to each genomic position
        sigList: a list of signals generated from one read.
        kmerWindow: length of kmers to fetch signals.
        signalWindow: number of signals to feed into neural network.
    '''
    #### Explanation of how signals are fetched according to genomic position of kmer ####
    ## genome: ACccgttagctaTAAACGTA, siglenList = [4,10,12,19,29,69, 110, 129, 140, 168], kmerWindow = 5 ##
    ## sigLenList_startIdx = sigLenList_init+pos-1 = 4, sigLenList_endIdx = pos_sigLenList_start+kmerWindow = 9 ##
    ## sigList_startIdx = int(siglenList[sigLenList_startIdx]) = 29, sigList_endIdx = int(siglenList[sigLenList_endIdx]) = 168 ##
    
    sigLenList_startIdx = start-1
    sigLenList_endIdx = end
    # print(sigLenList_startIdx, sigLenList_endIdx)
    # print(len(sigLenList))
    # if start=0
    if sigLenList_startIdx<0:
        sigList_startIdx = 0
    elif sigLenList_startIdx >= len(sigLenList):
        return 'end'
    else:
        sigList_startIdx = int(sigLenList[sigLenList_startIdx])
    
    if sigLenList_endIdx >= len(sigLenList):
        sigLenList_endIdx = -1
    sigList_endIdx = int(sigLenList[sigLenList_endIdx])

    # if no signals aligned to this kmer
    if sigList_startIdx == sigList_endIdx:
        return 'del'
    
    # reached the end of the signal list
    # sigList = [0,1,2,3,...,11,12,13,14,15], signalWindow = 5, start = 12, len(sigList) = 16
    # len(sigList)-end < signalWindow does not matter because python automatically clipps
    if len(sigList)-sigList_startIdx < signalWindow:
        return 'end'
    
    signals = [float(s) for s in sigList[sigList_startIdx:sigList_endIdx]]
    
    # start and end is too close that there are less than 400 signals
    if len(signals) < signalWindow:
        return 'del'
        
    return signals