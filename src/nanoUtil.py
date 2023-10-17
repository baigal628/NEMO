from seqUtil import *

def parseEventAlign(eventAlign = '', outfile = '', readname = '', chr_region = '', print_sequence = False, n_rname = 0, header = True):
    '''
    This function reads nanopolish eventalign file, aggregates signals and the number of 
    signals correspinding to one base movement for read in readname list.
    
    input:
        eventAlign: nanopolish eventalign output file.
        readname: A list containing readnames.
        chr_region: chromosome number that region of interest falls in.
    optional:
        print_sequence: if True, kmer sequence will be included in outfile.
        n_rname: number of readnames can be skipped in the readname list (default: 0).
                 Searching all the readnames from the eventalign file takes longer time.
    output: 
        outfile: siganlAlign.tsv with format: readname\tchrom\teventStart(reference)\tsigList\tsigLenLsit

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

            if thischrom != chr_region:
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
                out = "{}\t{}\t{}\t{}\t{}\t{}\n".format(read, chrom, eventStart, sequence, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
            else:
                out = "{}\t{}\t{}\t{}\t{}\n".format(read, chrom, eventStart, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
            if outfile:
                outf.write(out)
    outf.close()

def parseSigAlign(sigAlign):
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
            sigList = line[3].split(',')
            siglenList = line[4].split(',')
            
            # This will output within prange, the 80bp kmer window with it's scores
            yield(readID, eventStart, sigList, siglenList, )