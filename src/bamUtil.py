import pysam
import os
from seqUtil import fetchSize, compliment
import numpy as np

def getAlignedReads(bam, region, genome, print_quer = False, print_name = False, refSeq = False, print_ref = False,
                    print_align = False, compliment_reverse = True, include_quer = False, qual = 15):
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
        compliment_reverse: Set True to compliment reads mapped to reverse strands.
        include_quer: include the read sequence in the out dictionary

    Output:
        out: a python dictionary with readname as key and chrom, alignstart, alignend, strand as value.
        rchrom: query chromsome name. A string if single position is specified. A list of chromsome number if region is set as 'all'.
        rqstart: query start position. A string if single position is specified. A list of chromsome number if region is set as 'all'.
        rqend: query end position. A string if single position is specified. A list of chromsome number if region is set as 'all'.
    '''
    print('Collecting reads mapped to ', region, ' ...')
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
            if  s.is_mapped and not s.is_secondary and not s.is_supplementary:
                if np.mean(s.query_qualities) < qual: continue
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
                if compliment_reverse:
                    if strand == -1:
                        alignedRead = compliment(alignedRead)
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
    print(len(out), " reads in total.")
    return out, rchrom, rqstart, rqend


def readstoIdx(outpath, prefix, region, bam = '', ref = '', reads=''):
    '''
    fetch reads mapped to region, create alignmen object and read to idx tsv file.
    
    '''
    # fetch reads based on genomic alignment
    outfile = os.path.join(outpath, prefix + '_readID.tsv')

    alignment, chrom, qStart, qEnd = getAlignedReads(bam, region, ref)
    # readDict = {readname: (readID, strand)}
    readDict = {r:(i, alignment[r][3]) for r,i in zip(alignment, range(len(alignment)))}
    
    readFh = open(outfile, 'w')
    readFh.write('readname\tread_id\tstrand\n')
    for k,v in readDict.items(): readFh.write(f'{k}\t{v[0]}\t{v[1]}\n')
    readFh.close()
    
    print(len(readDict), " reads in total.")
    
    return readDict

def idxToReads(bam, region, ref, readID):
    '''
    Given indexed reads as readID.tsv file, fetch reads mapped to the region and return read idx.
    '''
    
    print('readling read list...')
    readsToIdx = {}
    with open(readID, 'r') as infile:
        header = infile.readlines(1)
        for line in infile:
            line = line.strip().split('\t')
            # readname: line[0] idx: line[1]
            readsToIdx[line[0]] = line[1]
    alignment, chrom, start, end = getAlignedReads(bam, region, ref)
    myreads = {readsToIdx[r]:(r, alignment[r][3]) for r in alignment}
    
    return myreads, chrom, start, end


def modBamtoPred(modbam, region=''):
    typesOfMods = {'5mC':[('C', 0, 'm')], '5hmC': [('C', 0, 'h')], '5fC': [('C', 0, 'f')], '5caC': [('C', 0, 'c')],
               '5hmU': [('T', 0, 'g')], '5fU': [('T', 0, 'e')], '5caU': [('T', 0, 'b')],
               '6mA': [('A', 0, 'a'), ('A', 0, 'Y')], '8oxoG': [('G', 0, 'o')], 'Xao': [('N', 0, 'n')]}
    
    
    compbase = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N'}
    def getcomp(seq):
        newseq = []
        for base in seq: newseq.append(compbase[base])
        return ''.join(newseq)#newseq[::-1]
        
    pred={}
    samfile = pysam.AlignmentFile(modbam, "rb")
    
    if region:
        chrom = region.split(':')[0]
        if '-' in region:
            locus = region.split(':')[1].split('-')
            qstart, qend = int(locus[0]), int(locus[1])
        reads = samfile.fetch(chrom, qstart, qend)
    else:
        reads = samfile
    
    for s in reads:
        chr = s.reference_name
        if chr not in pred:
            pred[chr] = {}
        if s.is_mapped and not s.is_supplementary and not s.is_secondary:
            alignstart, alignend = s.reference_start, s.reference_end
            readname = s.query_name
            cigar = s.cigartuples
            if not cigar:
                continue
            strand = -1 if s.is_reverse else 1
            if (readname, strand) not in pred[chr]:
                pred[chr][(readname, strand)] = {}
            posstag = typesOfMods['6mA']
            if s.is_reverse: posstag = [(x[0], 1, x[2]) for x in posstag]
            ml = None
            for t in posstag:
                if t in s.modified_bases:
                    ml = s.modified_bases[t]
                    break
            if not ml:
                print(readname, 'does not have modification information', s.modified_bases.keys())
                continue

            if s.has_tag('MM'):
                skippedBase = -1 if s.get_tag('MM').split(',', 2)[0][-1] == '?' else 0
            elif s.has_tag('Mm'):
                skippedBase = -1 if s.get_tag('Mm').split(',', 2)[0][-1] == '?' else 0
            else:
                continue

            seq = s.query_sequence
            seqlen = len(seq)
            if s.is_reverse:  ###need to get compliment of sequence, but not reverse!!
                seq = getcomp(seq)

            seqApos = []
            c = 0
            for b in seq:
                if b == 'A':
                    seqApos.append(c)
                c += 1

            ml = dict(ml)
            for i in seqApos:
                if i not in ml:
                    ml[i] = skippedBase

            ref, quer = 0, 0
            for block in cigar:
                if block[0] in {0, 7, 8}:  # match, consumes both
                    for i in range(block[1]):
                        if quer in ml: pred[chr][(readname, strand)][ref + alignstart] = ml[quer]
                        ref += 1
                        quer += 1
                elif block[0] in {1, 4}:  # consumes query
                    quer += block[1]
                elif block[0] in {2, 3}:  # consumes reference
                    ref += block[1]
            dirtowrite = '-' if s.is_reverse else '+'
    return pred