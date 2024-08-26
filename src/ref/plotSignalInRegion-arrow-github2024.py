import matplotlib.pyplot as plt
import sys, os, struct
import pysam
import numpy as np
from statistics import median, stdev
import pyarrow as pa
import pyarrow.parquet as pq
import time

"""
python /private/groups/brookslab/cafelton/fusions-code/pod5kmersignal/plotSignalInRegion.py chr13:52310695-52310705
        can_mappings.reform-compressed-readnameToCode.tsv can_mappings.bam can_mappings.reform-compressed-kmersignal.bin 
        mod_mappings.reform-compressed-readnameToCode.tsv mod_mappings.bam mod_mappings.reform-compressed-kmersignal.bin 
"""

# python /private/groups/brookslab/cafelton/fusions-code/pod5kmersignal/plotSignalInRegion.py chr13:52310695-52310705 can_mappings.reform-compressed-readnameToCode.tsv can_mappings.bam can_mappings.reform-compressed-kmersignal.bin mod_mappings.reform-compressed-readnameToCode.tsv mod_mappings.bam mod_mappings.reform-compressed-kmersignal.bin

#python3 /mnt/hdd/nanopore2/viz/pod5-to-kmer-signal-main/plotSignalInRegion.py chrI:100000-100020 nuc_shiftscale_movesout.sorted-scaled-readnameToCode.tsv nuc_shiftscale_movesout.sorted.bam nuc_shiftscale_movesout.sorted-scaled-kmersignal.tsv


compbase = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N'}
def getrevcomp(seq):
    seq = seq.upper()
    newseq = []
    for base in seq: newseq.append(compbase[base])
    return ''.join(newseq[::-1])


# readconvfile = sys.argv[2]
# bamfile = sys.argv[3]
# sigfile = sys.argv[4]

region = sys.argv[1]
# region = 'chr13:52310500-52310515' #'chr13:52310695-52310705' #'chr13:52308762-52308776'   #'chr13:52309895-52309915'
qstrand = -1
# modpos = 8

qchr, temp = region.split(':')
qstart, qend = temp.split('-')
qstart, qend = int(qstart), int(qend)

allfiles = []
for i in range(2, len(sys.argv), 2):
    print(len(sys.argv), i)
    allfiles.append([sys.argv[i], sys.argv[i+1]])


allreadtosig = []
for z in allfiles:#[[readconvfile, bamfile, sigfile], [sys.argv[5], sys.argv[6], sys.argv[7]]]:
    bamfile, sigfile = z[0], z[1]

    readtoseq = {}
    regionrefseq = None
    samfile = pysam.AlignmentFile(bamfile, "rb")

    
    for s in samfile.fetch(qchr, qstart, qend):
        alignstart, alignend = s.reference_start, s.reference_end
        readname = s.query_name
        #print(qchr, alignstart, alignend, readname)
        if alignstart <= qstart and alignend >= qend: #and readname in processedreads:
            
            strand = -1 if s.is_reverse else 1
            chr = s.reference_name
            seq = s.query_sequence#s.get_reference_sequence() #s.query_sequence ####UNSURE ABOUT THIS TBH
                ###Needs to be query sequence, but need to calculate query alignment to reference
            if not regionrefseq: regionrefseq = s.get_reference_sequence()[qstart-alignstart:qend-alignstart]
            if s.is_reverse: seq = getrevcomp(seq)

            ref, quer = alignstart, 0
            queryPosToRef = {}

            readtoseq[readname] = [chr, strand, alignstart, seq, queryPosToRef]

    print(len(readtoseq))

    # start = time.time()
    # readtosiginregion = {}
    # parquet_file = pq.ParquetFile(sigfile)
    # print(parquet_file.num_row_groups)
    # #['readcode','kmeridx', 'qkmer', 'refpos', 'refkmer', 'signalLen', 'signal']
    #
    # #for batch in parquet_file.iter_batches(batch_size=10000):
    # c = 0
    # for z in range(parquet_file.num_row_groups):
    #     batch = parquet_file.read_row_group(z)
    #     print(z)
    # # for batch in parquet_file.iter_batches():
    #     #print(len(pqnames))
    #     # for i in range(batch.num_rows):
    #     #     # if c == 0:
    #     #     #     print(row)
    #     #     #     print(row['readcode'][0], type(row['readcode'][0]), row['readcode'][0].as_py(), type(row['readcode'][0].as_py()))
    #     #     # c += 1
    #     #     readname = batch['readcode'][i].as_py()
    #     #     pqnames.add(readname)
    #     #     if readname in readtoseq:
    #     #         thissig = batch['signal'][i].as_py()
    #     #         refpos = batch['refpos'][i].as_py()
    #     #         qkmer = batch['qkmer'][i].as_py()
    #     #         strand = readtoseq[readname][1]
    #     #         if qstrand == strand:
    #     #             if readname not in readtosiginregion: readtosiginregion[readname] = {}
    #     #             readtosiginregion[readname][refpos] = [qkmer[4]] + thissig
    #
    #
    #     #print('batch started', z)
    #     ##c = 0
    #     for row in zip(*batch.columns):
    #        readname = row[0].as_py()
    #        #pqnames.add(readname)
    #        # if c <1: print(readname, readname == t, type(readname))
    #        # c += 1
    #        if readname in readtoseq:
    #            thissig, refpos, qkmer = row[6].as_py(), row[3].as_py(), row[2].as_py()
    #            strand = readtoseq[readname][1]
    #            #if readname not in readtosiginregion: print(readname, strand)
    #            if qstrand == strand:
    #                if readname not in readtosiginregion: readtosiginregion[readname] = {}
    #                readtosiginregion[readname][refpos] = [qkmer[4]] + thissig
    #
    # print(time.time()-start)

    start = time.time()
    readtosiginregion = {}
    parquet_file = pq.ParquetFile(sigfile)
    print(parquet_file.num_row_groups)
    # ['readcode','kmeridx', 'qkmer', 'refpos', 'refkmer', 'signalLen', 'signal']

    # for batch in parquet_file.iter_batches(batch_size=10000):
    c = 0
    for z in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(z)
        print(z)
        # for batch in parquet_file.iter_batches():
        # print(len(pqnames))
        for i in range(batch.num_rows):
            # if c == 0:
            #     print(row)
            #     print(row['readcode'][0], type(row['readcode'][0]), row['readcode'][0].as_py(), type(row['readcode'][0].as_py()))
            # c += 1
            readname = batch['readcode'][i].as_py()
            if readname in readtoseq:
                thissig = batch['signal'][i].as_py()
                refpos = batch['refpos'][i].as_py()
                qkmer = batch['qkmer'][i].as_py()
                strand = readtoseq[readname][1]
                # if qstrand == strand:
                if readname not in readtosiginregion: readtosiginregion[readname] = {}
                readtosiginregion[readname][refpos] = [qkmer[4],strand] + thissig

        # # print('batch started', z)
        # ##c = 0
        # for row in zip(*batch.columns):
        #     readname = row[0].as_py()
        #     # pqnames.add(readname)
        #     # if c <1: print(readname, readname == t, type(readname))
        #     # c += 1
        #     if readname in readtoseq:
        #         thissig, refpos, qkmer = row[6].as_py(), row[3].as_py(), row[2].as_py()
        #         strand = readtoseq[readname][1]
        #         # if readname not in readtosiginregion: print(readname, strand)
        #         if qstrand == strand:
        #             if readname not in readtosiginregion: readtosiginregion[readname] = {}
        #             readtosiginregion[readname][refpos] = [qkmer[4]] + thissig
    print(time.time() - start)

    
    #for line in open(sigfile):
    #    c, kmeridx, qkmer, refpos, refkmer, signallen, thissig = line.rstrip().split('\t')
    #    if int(c) in codetoread:
    #        readname = codetoread[int(c)]
    #        if readname in readtoseq:
    #            thissig = [float(x) for x in thissig.split(',')]
    #            kmeridx, refpos, signallen = int(kmeridx), int(refpos), int(signallen)
    #            chr, strand, alignstart, seq, queryPosToRef = readtoseq[readname]
    #            if qchr == chr and qstrand == strand: #and kmeridx in queryPosToRef: #qstart <= kmeridx + alignstart <= qend:
    #                if readname not in readtosiginregion: readtosiginregion[readname] = {}#{'sig':[], 'seq': ''}
    #                readtosiginregion[readname][refpos] = [qkmer[4]] + thissig
    # print(readtosiginregion)
    print(len(readtosiginregion.keys()))
    allreadtosig.append(readtosiginregion)

print('files processed, plotting')

###non-overlapping plots
# fig, axs = plt.subplots(len(readtosiginregion))
# c = 0
# # print(readtosiginregion)
# for r in readtosiginregion:
#     linepos = 0
#     for p in range(qstart, qend):
#         if p in readtosiginregion[r]:
#             # print(r, p, readtosiginregion[r][p][0], len(readtosiginregion[r][p])-1)
#             siglen = len(readtosiginregion[r][p])-1
#             # axs[c].axvline(linepos + siglen, c='black', linewidth=1)
#             axs[c].text(linepos + siglen / 2, 100, readtosiginregion[r][p][0], ha='center')
#             axs[c].plot(list(range(linepos, siglen+linepos)), readtosiginregion[r][p][1:])
#         else:
#             axs[c].plot(list(range(linepos, 40+linepos)), [70]*40)
#         linepos += 40#siglen
#     c += 1


#print(allreadtosig)
fig, axs = plt.subplots(2)
fig.set_size_inches(6,8)
d = 0
sigdiff = [[] for x in range(qend-qstart)]
for readtosiginregion in allreadtosig:
    c = 0
    avgsig = [[] for x in range(qend-qstart)]
    for r in readtosiginregion:
        linepos = 0
        for p in range(qstart, qend):
            if p in readtosiginregion[r] and regionrefseq[p-qstart] == readtosiginregion[r][p][0].upper(): #compbase[readtosiginregion[r][p][0].upper()]: #
                # print(r, p, readtosiginregion[r][p][0], len(readtosiginregion[r][p])-1)
                siglen = len(readtosiginregion[r][p])-2
                # axs[c].axvline(linepos + siglen, c='black', linewidth=1)
                xaxis = list(np.linspace(linepos+qstart, linepos + 1 + qstart, siglen + 1))[:-1]
                # axs[c].text(linepos + siglen / 2, 100, readtosiginregion[r][p][0], ha='center')
                if readtosiginregion[r][p][1] == 1:
                    axs[0].plot(xaxis, readtosiginregion[r][p][2:], c = 'C' + str(d), alpha=0.2)
                else: axs[1].plot(xaxis, readtosiginregion[r][p][2:], c = 'C' + str(d), alpha=0.2)
                if c == 0:
                    axs[0].text(qstart + linepos + .5, 60, regionrefseq[p-qstart], ha='center')
                avgsig[p-qstart].append(sum(readtosiginregion[r][p][1:])/siglen)
            linepos += 1#siglen
        c += 1
    # for p in range(qend-qstart):
    #     plt.plot([p+qstart, qstart+p+1], [median(avgsig[p])]*2, c='C' + str(d), alpha=1)
    #     # plt.text(qstart + p + .5, 50 + d * 80 + ((p%2)*5), str(round(stdev(avgsig[p]))), ha='center', c='C' + str(d))
    #     sigdiff[p].append(stdev(avgsig[p]))
    d += 1
# if len(sys.argv) >= 6:
#     for p in range(qend-qstart):
#         plt.text(qstart + p + .5, 130, str(round(abs(sigdiff[p][1]-sigdiff[p][0]))), ha='center')
axs[0].set_title('Forward strand')
axs[1].set_title('Reverse strand')
axs[0].ticklabel_format(useOffset=False, style='plain')
axs[1].ticklabel_format(useOffset=False, style='plain')
plt.savefig(bamfile.split('/')[-1].split('.')[0] + region + '-readsigplots-overlaid-with-variability-negstrand-arrow-new-new-new.png', dpi=600)
print('done')
