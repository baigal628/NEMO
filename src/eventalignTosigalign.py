import argparse
import os
from bamUtil import getAlignedReads, readstoIdx
from seqUtil import fetchSize

def reverseSigLenList(sigLenList):
    '''
    reverse 3'-[5, 12, 20]-5' to 5'-[8, 15, 20]-3' 
    '''
    
    sigLenList = sigLenList[::-1]
    newlist = []
    current = 0
    
    for i in range(len(sigLenList)-1):
        val = (sigLenList[i]-sigLenList[i+1]) + current
        newlist.append(val)
        current = val
    newlist.append(current+sigLenList[i+1])
    
    return newlist

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
    outf.write('read_id\tstrand\tchr\t5_end\tsigList\tsigLenList\n')
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
                print('skipping read ', thisread)
                continue
            thisstrand = reads[thisread][1]
            # reverse signal order if sequencing reverse strand
            raw_sig = line[-1].split(',') if thisstrand == 1 else line[-1].split(',')[::-1]
            # start of the new read
            if thisread != read:
                # not the very first read
                if sequence:
                    # reverse full sigList if reads from reverse strand so that
                    if strand == -1:
                        eventStart = eventStart+len(sigLenList)-1
                        # print(f'read {read} is on reverse strand, reversing signals')
                        sigList = sigList[::-1]
                        sigLenList = reverseSigLenList(sigLenList)
                    if print_sequence:
                        out = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(reads[read][0], reads[read][-1], chrom, eventStart, sequence, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
                    else:
                        out = "{}\t{}\t{}\t{}\t{}\t{}\n".format(reads[read][0], reads[read][-1], chrom, eventStart, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
                    outf.write(out)
                    # Set variables back to initial state
                    read = ''
                    sequence = ''
                    sigList = []
                    sigLenList = []
                
                # store current read
                read = thisread
                chrom = thischrom
                strand = thisstrand
                eventStart = int(line[1])
                start = line[1]
                kmer = line[2]

                # signals are stored in column 13/15 and are separated my comma
                sigList = [float(i) for i in raw_sig]
                sigLen = len(sigList)
                sigLenList = [sigLen]
                sequence = kmer
            
            # next kmer within the same read
            else:
                signals = [float(i) for i in raw_sig]
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
            if strand == -1:
                eventStart = eventStart+len(sigLenList)-1
                # print(f'read {read} is on reverse strand, reversing signals')
                sigList = sigList[::-1]
                sigLenList = reverseSigLenList(sigLenList)
            if print_sequence:
                out = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(reads[read][0], reads[read][-1], chrom, eventStart, sequence, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
            else:
                out = "{}\t{}\t{}\t{}\t{}\t{}\n".format(reads[read][0], reads[read][-1], chrom, eventStart, ','.join(str(i) for i in sigList), ','.join(str(i) for i in sigLenList))
            outf.write(out)
    outf.close()

def splitSigalign(sigalign, ref, outpath, prefix):

    gsize = fetchSize(ref)
    outFh = {}
    for chr in gsize:
        outFile = os.path.join(outpath, prefix + '_' + str(chr) + '_sigalign.tsv')
        outFh[str(chr)] = open(outFile, 'w')
    with open(sigalign, 'r') as sigFh:
        for line in sigFh:
            chrom = line.strip().split('\t', 3)[1]
            outFh[chrom].write(line)
    for chr in gsize:
        outFh[chr].close()

def add_parser(parser):
    parser.add_argument('--bam', type = str, default='', help = 'coordinate sorted, indexed bam file.')
    parser.add_argument('--ref', type = str, default='', help = 'refence genome.')
    parser.add_argument('--eventalign', type = str, default='', help = 'eventalign file.')
    parser.add_argument('--outpath', type = str, default='./', help = 'output path.')
    parser.add_argument('--prefix', type = str, default='', help = 'outfile prefix.')
    parser.add_argument('--reads', nargs="*", default=[], help = 'a list of reads to parse signals.')
    parser.add_argument('--region', type = str, default='', help = '')
    parser.add_argument('--split_sig', action='store_true', help = 'add the tag to split sigalign output by chromsome.')
    parser.add_argument('--header', action='store_true', help = 'add the tag to indicate there is header in eventalign file.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='eventalign to sigalign file.')           
    add_parser(parser)
    args = parser.parse_args()
    reads = readstoIdx(outpath = args.outpath, prefix = args.prefix, bam = args.bam, ref = args.ref, region = args.region, reads = args.reads)
    parseEventAlign(args.eventalign, args.outpath, args.prefix, reads, header=args.header)
    if args.split_sig:
        outsig = os.path.join(args.outpath, args.prefix + '_sigalign.tsv')
        splitSigalign(outsig, args.ref, args.outpath, args.prefix)
    print('Done processing ', args.eventalign)