from seqUtil import fetchSize
import matplotlib.pyplot as plt
import numpy as np
from predict import aggregate_scores
import pysam

def gettss(bed, genome, window, col = {'chrom':0, 'start':1, 'end':2, 'strand':5}):
    
    tssposs = {}
    hw = window/2
    genomeSize = fetchSize(genome)
    with open(bed, 'r') as infile:
        for line in infile:
            if 'track' not in line:
                line = line.strip().split('\t')
                if len(line) == 1:
                    continue
                chr, dir = line[col['chrom']], line[col['strand']]
                pos = int(line[col['start']]) if dir == '+' else int(line[col['end']])
                if chr not in tssposs:
                    tssposs[chr] = []
                left=int(max(0, pos-hw))
                right=int(min(pos+hw, genomeSize[chr]))
                tssposs[chr].append((left, pos, right, dir))
        return tssposs

def plotmetagene(predout, bed, genome, window, method, sw='', space=150, labels=('distance to tss (bp)', 'prediction score'), 
                 thred = '', outpath='', prefix = '', color = 'tab:blue', legend = '', odd=False, ylim = (0,1), alpha=0.4, 
                 return_value=False, strandness = True, bed_col = {'chrom':0, 'start':1, 'end':2, 'strand':5}):
    
    tssposs = gettss(bed=bed, genome=genome, window=window, col = bed_col)

    hw = window/2
    all_tss_scores = []
    
    if not isinstance(predout, list):
        predout = [predout]
        color = [color]
        legend = [legend]
    for i in range(len(predout)):
        tssscores = [[] for i in range(window+1)]
        pred = predout[i]
        for chrom, reads in pred.items():
            if chrom not in tssposs:
                print(chrom, ' not in input bed.')
                continue
            for read, read_pred in reads.items():
                if not read_pred:
                    continue
                sortedread = sorted(read_pred.items())
                thickStart, thickEnd = sortedread[0][0], sortedread[-1][0]
                for tsspos in tssposs[chrom]:
                    # has overlap
                    if thickStart < tsspos[0] and thickEnd > tsspos[2]:
                        for (pos, scores) in sortedread:
                            if pos>=tsspos[0] and pos<= tsspos[2]:
                                if strandness:
                                    repos = int(hw+pos-tsspos[1]) if tsspos[3] == '+' else int(tsspos[1]-pos+hw)
                                else:
                                    repos = int(hw+pos-tsspos[1])
                                score = aggregate_scores(scores, method[0])
                                if odd:
                                    score = 0.99 if score == 1 else score
                                    score = np.log(score/(1-score))
                                tssscores[repos].append(score)
        
        
        tssscores = [aggregate_scores(x, method[1], thred) if len(x) > 0 else 0 for x in tssscores]
        all_tss_scores.append(tssscores)
        print(len(all_tss_scores))
    

    plt.figure(figsize=(6,4))
    for i in range(len(all_tss_scores)):
        tssscores = all_tss_scores[i]
        print(len(tssscores))
        if sw:
            hsw = int(round(sw/2))
            xval, yval = [], []
            for j in range(hsw, (window+1)-(hsw+1), int(round(hsw/2))):
                thesescores = tssscores[j-hsw:j+hsw]
                avg = sum(thesescores)/len(thesescores)
                yval.append(avg)
                xval.append(j-hw)
        else:
            yval = tssscores
            xval = np.arange(-hw, hw+1)
        plt.plot(xval, yval, color=color[i], label=legend[i], alpha=alpha)
    
    plt.xticks(np.concatenate((np.flip(np.arange(0, -hw-1, -space)[1:]), np.arange(0, hw+1, space)), axis=0), rotation='vertical')
    plt.grid(alpha=0.5,axis = 'x')
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    plt.savefig(outpath+prefix+'_aggregate.pdf', dpi = 200)
    plt.show()
    plt.close()
    if return_value:
        return all_tss_scores

def plotDistribution(predout, outpath, prefix, method, legend, color):
    
    if not isinstance(predout, list):
        predout = [predout]
        color = [color]
        legend = [legend]

    preds = []
    for i in range(len(predout)):
        pred = []
        for chrom, reads in predout[i].items():
            for read, read_pred in reads.items():
                for pos, scores in read_pred.items():
                    pred.append(aggregate_scores(scores, method[0]))
        print('mean: ', np.mean(pred))
        preds.append(pred)
    
    for i in range(len(preds)):
        plt.hist(preds[i], bins=100, alpha=0.4, label = legend[i], color=color[i])
    
    plt.title(prefix)
    plt.legend()
    plt.savefig(outpath+prefix+'_dist.pdf', dpi = 200)
    plt.show()
    plt.close()


def modBamtoPred(modbam):
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
    for s in samfile:
        chr = s.reference_name
        if chr not in pred:
            pred[chr] = {}
        if not s.is_secondary:
            alignstart, alignend = s.reference_start, s.reference_end
            readname = s.query_name
            cigar = s.cigartuples
            if not cigar:
                continue
            if readname not in pred[chr]:
                pred[chr][readname] = {}
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
                        if quer in ml: pred[chr][readname][ref + alignstart] = ml[quer]
                        ref += 1
                        quer += 1
                elif block[0] in {1, 4}:  # consumes query
                    quer += block[1]
                elif block[0] in {2, 3}:  # consumes reference
                    ref += block[1]
            dirtowrite = '-' if s.is_reverse else '+'
    return pred


def bedtoPred(bed):
    pred={}
    with open(bed) as infile:
        for line in infile:
            line=line.strip().split('\t')
            chr = line[0]
            if chr not in pred:
                pred[chr] = {'read':{}}
            astart = int(line[1])
            prob = float(line[3])
            pred[chr]['read'][astart] = prob
    return pred