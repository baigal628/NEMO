import pysam
import numpy as np
import random
from seqUtil import *

def getAlignedReads(sam, region, pname = False, reverse = False):
    '''
    sam: input sam/bam file.
    region: region to fetch aligned reads.
        E.g. region = chrI:12300-12500
    pname: print the readname or not.
    '''
    
    chrom = region.split(':')[0]
    locus = region.split(':')[1].split('-')
    qstart, qend = int(locus[0]), int(locus[1])
    qrange = qend - qstart
    
    samFile = pysam.AlignmentFile(sam)
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
                        alignedRead += cigarTag[1]*'N'
                    else:
                        quer += cigarTag[1]
                if pname:
                    print('>', s.query_name, alignstart, alignend)
                qpos = qstart-alignstart
                if reverse:
                    if s.is_reverse:
                        alignedRead = reverseCompliment(alignedRead)
                string = alignedRead[qpos:qpos+qrange]
                print(string)
    samFile.close()