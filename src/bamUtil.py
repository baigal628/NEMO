import pysam
from seqUtil import fetchSize, reverseCompliment

def getAlignedReads(bam, region, genome, print_quer = False, print_name = False, refSeq = False, print_ref = False,
                    print_align = False, reverse = False, include_quer = False):
    '''
    Input:
        bam: input sam/bam file.
                genome: reference genome indexed with faidx.
        region: region to fetch aligned reads.
            E.g. region = 'all' for whole genome or region = 'chrI:12300-12500' or region = 'chrV' for specific genome ranges.
        print_name: Set True to print readname.
        refSeq: Set True to store refernce sequences in the output.
        print_ref: Set True to print reference sequnce. If true, must provide genome.
        print_align: Set True to print align start and end of reads.
        reverse: Set True to reverse compliment reads mapped to negative strands.

    Output:
        out: a python dictionary with readname as key and chrom, alignstart, alignend, strand as value.
        rchrom: query chromsome name. A string if single position is specified. A list of chromsome number if region is set as 'all'.
        rqstart: query start position. A string if single position is specified. A list of chromsome number if region is set as 'all'.
        rqend: query end position. A string if single position is specified. A list of chromsome number if region is set as 'all'.
    '''
    
    cigarCode = {1: 'I', 2: 'D', 3: 'N'}
    regions = []
    # format is 'all'
    if region == 'all':
        rchrom, rqstart, rqend = [], [], []
        genomeSize = fetchSize(genome)
        for chrom, qend in genomeSize.items():
            qstart = 0
            rchrom.append(chrom)
            rqstart.append(qstart)
            rqend.append(qend)
            regions.append((chrom, qstart, qend))
    
    # format is 'chrIV' or 'chrII:3-2564'
    else:
        chrom = region.split(':')[0]
        if '-' in region:
            locus = region.split(':')[1].split('-')
            qstart, qend = int(locus[0]), int(locus[1])
        else:
            genomeSize = fetchSize(genome)
            qstart = 0
            qend = genomeSize[chrom]
        rchrom = chrom
        rqstart = qstart
        rqend = qend
        regions.append((chrom, qstart, qend))
    
    qrange = qend - qstart
    out = {}
    samFile = pysam.AlignmentFile(bam, 'rb')

    if refSeq:
        refFile = pysam.FastaFile(genome)
        read = refFile.fetch(chrom, qstart, qend)
        if print_name:
            print(region)
        if print_ref:
            print(read)
        out['ref'] = read
    
    for (chrom, qstart, qend) in regions:
        for s in samFile.fetch(chrom, qstart, qend):
            if not s.is_secondary and not s.is_supplementary:
                alignstart, alignend = s.reference_start, s.reference_end
                if s.is_reverse:
                    strand = -1
                else:
                    strand = 1
                
                # if alignstart <= qstart and alignend >= qend:
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
                if reverse:
                    alignedRead = reverseCompliment(alignedRead)
                if print_quer:
                    qpos = qstart-alignstart
                    if qpos<0:
                        string = (alignstart-qstart)*'S' + alignedRead[:qrange]
                    else:
                        string = alignedRead[qpos:qpos+qrange]
                    print(string[:10])
                
                if print_align:
                    print(alignstart, alignend, strand)
                
                if include_quer:
                    out[s.query_name] = (chrom, alignstart, alignend, alignedRead, strand)
                else:
                    out[s.query_name] = (chrom, alignstart, alignend, strand)
        print('finshed fetching ', chrom, qstart, qend)
    samFile.close()
    return out, rchrom, rqstart, rqend