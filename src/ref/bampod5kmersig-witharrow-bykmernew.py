import matplotlib.pyplot as plt
import numpy as np
import pod5 as p5
import sys
import struct
import os
import math, statistics
import pysam
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

parser = argparse.ArgumentParser(description='Predicting the best modification threshold and dist from positive and negative control bam files. Will output a tsv file containing those values.',
                                 usage='python[3+] predictthreshold.py -p pos.bam -n neg.bam OR JUST -s sample.bam')
parser.add_argument('-b', '--bam', action='store', dest='b',
                    help='R10 basecalled bam file, should have moves table and be indexed. Can be filtered to locus or not. Only reads in this bam will have signal called for kmers')
parser.add_argument('-p', '--pod5', action='store', dest='p',
                    help='R10 pod5 file, should have all reads that are included in bam. This should be a single file, not multiple')
parser.add_argument('-o', '--outputprefix', action='store', dest='o',
                    help='Output filename prefix')
parser.add_argument('-s', '--scale', action='store_true', dest='s',
                    help='whether to scale the signal values to facillitate comparison between reads, requires that pod5 reads have a non-nan value in the predicted_scaling area')
parser.add_argument('-c', '--chr', action='store', dest='c',
                    help='[optional] chromosome to get reads from')
parser.add_argument('-n', '--numreads', action='store', dest='n',
                    help='[optional] number of reads to get. if -c is specified, will be first n reads on the chromosome specified, otherwise will be first n reads on first chr')

args = parser.parse_args()

samfile = pysam.AlignmentFile(args.b, "rb")
reader = p5.Reader(args.p)
all_reads = reader.read_ids
compbase = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N'}
def revcomp(seq):
    newseq = []
    for base in seq: newseq.append(compbase[base])
    return ''.join(newseq[::-1])

readchunks = {}
c, d = 0, 0
thischr, thisstart, thisend = '', 0, 0
thischunk = set()
for s in samfile:
    alignchr = s.reference_name
    if alignchr != 'chrM' and (not args.c or alignchr == args.c) and (not args.n or c < int(args.n)):
        if s.is_mapped and not s.is_supplementary and not s.is_secondary:
            if s.query_name not in all_reads: continue
            alignstart, alignend = s.reference_start, s.reference_end
            base_qualities = s.query_qualities
            if sum(base_qualities) / len(base_qualities) <= 10: continue
            if thischr == '': thischr = alignchr
            d += 1
            if thischr != alignchr or d % 1000 == 0:
                readchunks[(thischr, thisstart, thisend)] = thischunk
                readsadded = d%1000 if d%1000 > 0 else 1000
                print('added chunk of ' + str(readsadded) + ' reads on ' + thischr)
                thischunk = set()
                thischr, thisstart = alignchr, alignstart   
            thisend = alignend
            thischunk.add(s.query_name)
            c += 1
    else:
        break
readchunks[(thischr, thisstart, thisend)] = thischunk


print('done getting chunks of readnames')
if not args.o: outprefix = '.'.join(args.b.split('/')[-1].split('.')[:-1])
else: outprefix = args.o
if args.s: outprefix += '-scaled'

kmer_length = 9


c = 0

###THIS will only work with merged pod5 files
# columnnames = ['readcode','kmeridx', 'qkmer', 'refpos', 'refkmer', 'signalLen', 'signal']
# temprow = [['abcdefg'], [0], ['AAAATAAAA'], [10000], ['T'], [5], [[1.2, 1.6, -.2, .7, 1.9]]]
# temparrow = pa.RecordBatch.from_arrays(temprow, names=columnnames)

# columnnames = ['readname','chr', 'startpos', 'signals', 'siglenperkmer']
# temprow = [['abcdefg'], ['chr'], [0], [[1.2, 1.6, -.2, .7, 1.9]], [[1, 1, 5, 10, 20]]]
# temparrow = pa.RecordBatch.from_arrays(temprow, names=columnnames)
#
# with pq.ParquetWriter(outprefix + '-sigalign.parquet', temparrow.schema) as writer:
columnnames = ['readcode','kmeridx', 'qkmer', 'refpos', 'refkmer', 'signalLen', 'signal']
temprow = [['abcdefg'], [0], ['AAAATAAAA'], [10000], ['T'], [5], [[1.2, 1.6, -.2, .7, 1.9]]]
temparrow = pa.RecordBatch.from_arrays(temprow, names=columnnames)

with pq.ParquetWriter(outprefix + '-kmersignal-complete.parquet', temparrow.schema) as writer:
    for chunkpos in readchunks:
        chunkdata = [[] for x in range(len(columnnames))]
        print(chunkpos, len(readchunks[chunkpos]))#, list(readchunks[chunkpos])[:5])
        readtoseq = {}
        for s in samfile.fetch(chunkpos[0], chunkpos[1], chunkpos[2]):
            if s.is_mapped and not s.is_supplementary and not s.is_secondary:
                # alignstart = s.reference_start if not s.is_reverse else s.reference_end
                base_qualities = s.query_qualities
                if sum(base_qualities) / len(base_qualities) <= 10: continue
                readname = s.query_name
                alignstart = s.reference_start
                strand = -1 if s.is_reverse else 1
                if readname in readchunks[chunkpos]:
                    chr = s.reference_name
                    seq = s.query_sequence #if not s.is_reverse else s.get_forward_sequence()
                    # if s.is_reverse: seq = getrevcomp(seq)
                    # len_seq = len(seq) - kmer_length + 1  # to get the number of kmers
                    seqlenforrev = len(seq)-1
                    refseq = s.get_reference_sequence().upper()
                    queryrefpospairs = dict(s.get_aligned_pairs())

                    ns = int(s.get_tag("ns")) ##number of signals
                    ts = int(s.get_tag("ts")) ##signal start delay
                    mv = s.get_tag("mv") ##signal list
                    len_mv = len(mv)

                    stride = mv[0]

                    mvpos = 1
                    move_count = 0

                    start_signal_idx = ts
                    currsiglen = 0
                    # kmer_idx = 0
                    kmer_idx = 0 if strand == 1 else seqlenforrev
                    kmersigpos = []

                    ###making assumption that every base in the query seq represents a kmer and signal chunk
                    while mvpos < len_mv:
                        mv_val = mv[mvpos]
                        currsiglen += stride
                        if mv_val == 1 or mv_val == len_mv-1:
                            if kmer_idx in queryrefpospairs and queryrefpospairs[kmer_idx] != None and 4 <= kmer_idx < len(seq)-5:
                                # kmersigpos.append((start_signal_idx, start_signal_idx+currsiglen, queryrefpospairs[kmer_idx]))
                                if strand == -1:
                                    kmersigpos.append((start_signal_idx, start_signal_idx + currsiglen, len(seq) - kmer_idx, revcomp(seq[kmer_idx - 4:kmer_idx + 5]), queryrefpospairs[kmer_idx], refseq[queryrefpospairs[kmer_idx] - alignstart]))  # (kmer_idx, start_signal_idx, start_signal_idx+currsiglen))
                                else:
                                    kmersigpos.append((start_signal_idx, start_signal_idx + currsiglen, kmer_idx, seq[kmer_idx - 4:kmer_idx + 5], queryrefpospairs[kmer_idx], refseq[queryrefpospairs[kmer_idx] - alignstart]))  # (kmer_idx, start_signal_idx, start_signal_idx+currsiglen))

                            # kmersigpos.append((start_signal_idx, start_signal_idx+currsiglen)) #(kmer_idx, start_signal_idx, start_signal_idx+currsiglen))
                            start_signal_idx += currsiglen
                            currsiglen = 0
                            kmer_idx = kmer_idx + 1 if strand == 1 else kmer_idx - 1
                        mvpos += 1
                    # print(kmersigpos)
                    readtoseq[readname] = [chr, strand, s.reference_start, s.reference_end, kmersigpos]

        print('done processing bam file for chunk ', chunkpos)


        c = 0
        d = 0
        readcodes = []
        for read in reader.reads(readchunks[chunkpos]):
            signal = read.signal_pa
            if args.s:
                scale = read.predicted_scaling.scale
                shift = read.predicted_scaling.shift
                signal = [float((s-shift)/scale) for s in signal]
            else: signal = [float(x) for x in signal]
            readname = str(read.read_id)
            if readname in readtoseq:
                # outline = [readname, readtoseq[readname][0], readtoseq[readname][2], [], []]
                # lastsigtot = 0
                strand = readtoseq[readname][1]
                #laststart = readtoseq[readname][2] if strand == -1 else readtoseq[readname][3]
                # print(strand, len(readtoseq[readname][-1]))
                for sigstart, sigend, kmeridx, qkmer, refpos, refkmer in readtoseq[readname][-1]:
                    if strand == -1 and d < 10:
                        d += 1
                        print(sigstart, sigend, qkmer, kmeridx, refpos)
                    signalLen = sigend - sigstart
                    thissig = signal[sigstart:sigend]
                    if readtoseq[readname][1] == -1: thissig = thissig[::-1]
                    outdata = [readname, kmeridx, qkmer, refpos, refkmer, signalLen, thissig]  # ','.join(thissig)]
                    for j in range(7):
                        chunkdata[j].append(outdata[j])

        print('processed pod5 data for chunk ', chunkpos)
        arrowtable = pa.RecordBatch.from_arrays(chunkdata, names=columnnames)  # pa.Table.from_pydict(mydata)
        writer.write_batch(arrowtable)
        print('wrote arrow file for chunk ', chunkpos)
                

samfile.close()
