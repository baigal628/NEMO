def parseEventAlign(eventAlign = '', outfile = ''):
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
    
    outf = open(outfile, 'w')
    readID = ''
    sequence = ''
    signalLength = 0
    signalLengthList = []
    with open(eventAlign, 'r') as inFile:
        header = inFile.readline()
        for line in inFile:
            line = line.strip().split('\t')
            read = line[3]
            if readID != read:
                if sequence:
                    # Set variables back to initial state
                    line = "{}\t{}\t{}\t{}\t{}\t{}\n".format(readID, chrom, eventStart, sequence, ','.join(str(i) for i in signalList), ','.join(str(i) for i in signalLengthList))
                    outf.write(line)
                    readID = ''
                    sequence = ''
                    sigList = []
                    sigLength = 0
                    sigLengthList = []
                readID = read
                chrom = line[0]
                eventStart = line[1]
                start = line[1]
                kmer = line[2]
                # signals are stored in column 13 and are separated my comma
                signals = [float(i) for i in line[15].split(',')]
                signalList = signals
                signalLength = len(signals)
                sequence += kmer
                print(kmer)
            # next kmer within the same read
            else:
                signals = [float(i) for i in line[15].split(',')]
                # or signalList += signals
                signalList.extend(signals)
                # signalLength records the number of signals for one base movement
                signalLength += len(signals)
                # different kmer
                #  (kmer1, chrom1, start1) = (kmer0, chrom0, start0)
                if (line[1], line[2]) != (start, kmer):
                    start = line[1]
                    kmer = line[2]
                    sequence += kmer[-1]
                    signalLengthList.append(signalLength)
        if sequence:
            line = "{}\t{}\t{}\t{}\t{}\t{}\n".format(readID, chrom, eventStart, sequence, ','.join(str(i) for i in signalList), ','.join(str(i) for i in signalLengthList))
            outf.write(line)
    outf.close()


