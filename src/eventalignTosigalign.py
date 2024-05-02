import argparse
import os
from bamUtil import getAlignedReads

def add_parser(parser):
    parser.add_argument('--bam', type = str, default='', help = 'coordinate sorted, indexed bam file.')
    parser.add_argument('--ref', type = str, default='', help = 'refence genome.')
    parser.add_argument('--eventalign', type = str, default='', help = 'eventalign file.')
    parser.add_argument('--outpath', type = str, default='./', help = 'output path.')
    parser.add_argument('--prefix', type = str, default='', help = 'outfile prefix.')
    parser.add_argument('--reads', nargs="*", default=[], help = 'a list of reads to parse signals.')
    parser.add_argument('--region', type = str, default='', help = '')


def readstoIdx(outpath, prefix, bam = '', ref = '', region='', reads=''):
    # fetch reads based on genomic alignment
    outfile = os.path.join(outpath, prefix + '_readID.tsv')
    
    if region:
        alignment, chrom, qStart, qEnd = getAlignedReads(bam, region, ref)
        readDict = {r:i for r,i in zip(alignment, range(len(alignment)))}
    else:
        readDict = {r:i for r,i in zip(reads, range(len(reads)))}
    
    readFh = open(outfile, 'w')
    for k,v in readDict.items(): readFh.write('{read}\t{index}\n'.format(read = k, index = v))
    readFh.close()
    
    print(len(readDict), " reads in total.")
    
    return readDict

def parseEventAlign(eventAlign, outpath, prefix, reads, print_sequence = False, header = True):
    '''
    parseEventAlign collapse nanopolish eventalign to signal alignment file. Sigalign counts number of 
    signals corresponds to one base movement for single-read.
    Output sigalign file, a tsv with format: readname\tchrom\teventStart(reference)\tsigList\tsigLenList
        
    eventAlign: nanopolish eventalign output file.
    outpath: output file path
    prefix: output file prefix
    reads: list of reads to extract signals from.
    print_sequence: if True nucleic acid sequences for each read.
    header: whether include header in the output file.
    
 
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
    
    event_outfile = os.path.join(outpath, prefix + '_sigalign.tsv')

    outf = open(event_outfile, 'w')
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
            if reads:
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
            outf.write(out)
    outf.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='eventalign to sigalign file.')           
    add_parser(parser)
    args = parser.parse_args()
    reads = readstoIdx(outpath = args.outpath, prefix = args.prefix, bam = args.bam, ref = args.ref, region = args.region, reads = args.reads)
    parseEventAlign(args.eventalign, args.outpath, args.prefix, reads)
    print('Done processing ', args.eventalign)