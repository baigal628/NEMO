import pysam
import numpy as np
import random
from seqUtil import *

def getAlignedReads(sam, region, print_quer = False, print_name = False, print_ref = False, print_align = False, genome = '', reverse = False):
    '''
    sam: input sam/bam file.
    region: region to fetch aligned reads.
        E.g. region = chrI:12300-12500
    print_name: Set True to print readname.
    print_ref: Set True to print reference sequnce. If true, must provide genome.
    print_align: Set True to print align start and end of reads.
    genome: reference genome indexed with faidx.
    reverse: Set True to reverse compliment reads mapped to negative strands.
    '''
    
    cigarCode = {1: 'I', 2: 'D', 3: 'N'}
    chrom = region.split(':')[0]
    locus = region.split(':')[1].split('-')
    qstart, qend = int(locus[0]), int(locus[1])
    qrange = qend - qstart
    out = {}

    samFile = pysam.AlignmentFile(sam)
    if genome:
        refFile = pysam.FastaFile(genome)
        read = refFile.fetch(chrom, qstart, qend)
        if print_name:
            print(region)
        if print_ref:
            print(read)
        out['ref'] = read
    for s in samFile.fetch(chrom, qstart, qend):
        if not s.is_secondary and not s.is_supplementary:
            alignstart, alignend = s.reference_start, s.reference_end
            if alignstart <= qstart and alignend >= qend:
                seq = s.query_sequence
                # cigar is relative to the query sequence, which reflects the bases in reference seq.
                # Thus, don't do reverse compliment before extracting the mapped reads
                c = s.cigar
                quer = 0
                alignedRead = ''
                for cigarTag in c:
                    if cigarTag[0] in {0,7,8}:
                        alignedRead += seq[quer:quer+cigarTag[1]]
                        quer += cigarTag[1]
                    elif cigarTag[0] in {2,3}:
                        alignedRead += cigarTag[1]*cigarCode[cigarTag[0]]
                    elif cigarTag[0] in {1,4,5,6}:
                        quer += cigarTag[1]
                if print_name:
                    print(s.query_name)
                qpos = qstart-alignstart
                if reverse:
                    if s.is_reverse:
                        alignedRead = reverseCompliment(alignedRead)
                string = alignedRead[qpos:qpos+qrange]
                if print_quer:
                    print(string)
                if print_align:
                    print(alignstart, alignend)
                out[s.query_name] = string
    return out
    samFile.close()