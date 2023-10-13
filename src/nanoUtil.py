from seqUtil import *

def parseEventAlign(eventAlign = '', outfile = '', readname = '', chrom = '', genome = '', print_sequence = False, n_rname = 0):
    '''
    This function reads nanopolish eventalign file, aggregates signals and the number of 
    signals correspinding to one base movement for read in readname list.
    
    input:
    output: _siganlAlign.tsv with format: readname\tchrom\teventStart(reference)\tsigList\tsigLenLsit

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
    readname = set(readname)
    if outfile:
        outf = open(outfile, 'w')
    tag = ''
    read = ''
    sequence = ''
    c = 0
    readok = False
    
    with open(eventAlign, 'r') as inFile:
        header = inFile.readline()
        for line in inFile:
            line = line.strip().split('\t')
            thisread = line[3]
            thischrom = line[0]
            
            c+=1
            if c%10000000 == 0:
                print(c/1000000, ' M lines have passed.')

            if thischrom != chrom:
                continue

            # all line passed the chromosome and readname check should start here
            if thisread != read:
                if sequence:
                    # Set variables back to initial state
                    if print_sequence:
                        out = "{}\t{}\t{}\t{}\t{}\t{}\n".format(read, chrom, eventStart, sequence, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
                    else:
                        out = "{}\t{}\t{}\t{}\t{}\n".format(read, chrom, eventStart, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
                    if outfile:
                        outf.write(out)
                    if len(readname) <= n_rname:
                        sequence = ''
                        break
                    read = ''
                    sequence = ''
                    sigList = []
                    sigLenList = []
                
                if thisread in readname:
                    readname.remove(thisread)
                    print(len(readname), ' reads left in readname list')
                else:
                    continue
                
                # start new read here
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

                # check if this is a different kmer ot the same kmer
                # line[1]: this kmer start, line[2]: this kmer
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
        if sequence:
            if print_sequence:
                out = "{}\t{}\t{}\t{}\t{}\t{}\n".format(read, chromsome, eventStart, sequence, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
            else:
                out = "{}\t{}\t{}\t{}\t{}\n".format(read, chromsome, eventStart, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
            if outfile:
                outf.write(out)
    outf.close()

def parseSigAlign(sigAlign, pStart, pEnd, kmerWindow = 80):
    '''
    This function is a iterator that reads _siganlAlign.tsv file, and output readID sigList, sigLenList, and sigStart

    pStart: start position of region on reference genome
    pEnd: end position of region on reference genome
    sigStart: signal position 
    reference genome aligns to sigLenList[pStart-eventStart-1] on signalList.
    012345678
    accGTCGAa 
    sigList = [00,01,02,03,04,05,..]
    sigLenList = [12,32,51,71,96,26,136]
    GTCGA -> sigLen[51:51+400], sigLen[71:71+400] -> sigLenList[3-1]
    TCGAA -> sigLen[71:71+400], sigLen[96:96+400]
    '''

    with open(sigAlign) as sigAlignFh:
        for line in sigAlignFh:
            line = line.strip().split('\t')
            readID = line[0]
            eventStart = int(line[2])
            prange= pEnd-pStart
            sigList = line[3].split(',')
            siglenList = line[4].split(',')
            
            # This will output within prange, the 80bp kmer window with it's scores
            sigStart = pStart - eventStart -1
            
            yield(readID, sigList, siglenList, sigStart)