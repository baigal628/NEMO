from seqUtil import *

def parseEventAlign(eventAlign = '', outfile = '', readname = '', chrom = '', genome = '', print_sequence = False):
    '''
    alignScore function: Input eventalign file, for each read, the function aggregates the number of signals aligned to events with one base movement.
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
    tag = ''
    readID = ''
    sequence = ''
    signalLength = 0
    signalLengthList = []
    c = 0
    
    chromOrder = getchromOrder(genome)
    print(chromOrder)
    
    with open(eventAlign, 'r') as inFile:
        header = inFile.readline()
        for line in inFile:
            if tag == 'stop':
                print(warn)
                sequence = ''
                break
            line = line.strip().split('\t')
            read = line[3]
            
            c+=1
            if c%1000000 == 0:
                print(c)
            
            if chrom:
                # This part is problematic if eventalign file is not ordered
                if line[0] != chrom:
                    print(line[0])
                    print(chrom)
                    if chromOrder[line[0]] < chromOrder[chrom]: 
                        continue 
                    elif chromOrder[line[0]] > chromOrder[chrom]:
                        warn = 'stop by chromsome'
                        tag = 'stop'
            
            if readID != read:
                if readname:
                    if read not in readname:
                        continue
                    else:
                        readname.remove(read)
                        print(readname)
                        print(len(readname))
                        if len(readname) == 0:
                            warn = 'stop by readname'
                            tag = 'stop'
                if sequence:
                    # Set variables back to initial state
                    if print_sequence:
                        out = "{}\t{}\t{}\t{}\t{}\t{}\n".format(readID, chromsome, eventStart, sequence, ','.join(str(i) for i in signalList), ','.join(str(i) for i in signalLengthList))
                    else:
                        out = "{}\t{}\t{}\t{}\t{}\n".format(readID, chromsome, eventStart, ','.join(str(i) for i in signalList), ','.join(str(i) for i in signalLengthList))
                    if outfile:
                        outf.write(out)
                    readID = ''
                    sequence = ''
                    sigList = []
                    sigLength = 0
                    sigLengthList = []
#                 print('parsing read: ', read)
                readID = read
                chromsome = line[0]
                eventStart = line[1]
                start = line[1]
                kmer = line[2]
                # signals are stored in column 13/15 and are separated my comma
                signals = [float(i) for i in line[-1].split(',')]
                signalList = signals
                signalLength = len(signals)
                sequence += kmer
            # next kmer within the same read
            else:
                signals = [float(i) for i in line[-1].split(',')]
                # or signalList += signals
                signalList.extend(signals)
                # signalLength records the number of signals for one base movement
                signalLength += len(signals)
                # different kmer
                if (line[1], line[2]) != (start, kmer):
                    deletion = int(line[1]) - int(start) - 1
                    # deal with deletion in eventalign file
                    if deletion > 0:
                        sequence += deletion*'D'
                        for i in range(deletion):
                            signalLengthList.append(signalLengthList[-1])
                    start = line[1]
                    kmer = line[2]
                    sequence += kmer[-1]
                    signalLengthList.append(signalLength)
        if sequence:
            if print_sequence:
                out = "{}\t{}\t{}\t{}\t{}\t{}\n".format(readID, chromsome, eventStart, sequence, ','.join(str(i) for i in signalList), ','.join(str(i) for i in signalLengthList))
            else:
                out = "{}\t{}\t{}\t{}\t{}\n".format(readID, chromsome, eventStart, ','.join(str(i) for i in signalList), ','.join(str(i) for i in signalLengthList))
            if outfile:
                outf.write(out)
                outf.close()