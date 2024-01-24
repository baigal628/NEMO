import numpy as np

def predToBedGraph(infile, chr, bins, step, thred=0.5, outpath='', prefix =''):
    
    scoreDict = {i:[] for i in range(len(bins))}
   
    with open(infile, 'r') as predFh:
        for line in predFh:
            line = line.strip().split('\t')
            readname = line[0]
            strand = line[1]
            start = int(line[2])
            if start > bins[-1]:
                continue
            probs = line[3].split(',')
            end = start + step*(len(probs)-1)
            if end < bins[0]:
                continue
            else:
                i = int((start-bins[0])/step)
                if i < 0:
                    probs = probs[-i:]
                    i = 0
                for prob in probs:
                    prob = float(prob)
                    if np.isnan(prob):
                        continue
                    score = 1 if prob > thred else 0
                    scoreDict[i].append(score)
                    i +=1
                    if i >= len(bins):
                        break
    outfile = open(outpath+prefix+'.bedgraph', 'w')
    
    for i in scoreDict:
        start = bins[i]
        end = start+step
        avescore = sum(scoreDict[i])/len(scoreDict[i])
        outfile.write(chr + '\t' + str(start) + '\t' + str(end) + '\t' + str(avescore) + '\n')
    
    outfile.close()


def reformatBedgraph(bdg):

    prefix = bdg.split('.bedgraph')[0]
    outFh = open(prefix+'_1bp.bedgraph', 'w')
    
    with open(bdg, 'r') as inputFh:
        for line in inputFh:
            line = line.strip().split('\t')
            chrom = line[0]
            start = int(line[1])
            end = int(line[2])
            value = float(line[3])
            # Convert the interval into 1 base pair intervals
            for pos in range(start, end):
                outFh.write('{chrom}\t{start}\t{end}\t{value}\n'.format(chrom = chrom, start = pos, end = pos+1, value = value))
    
    outFh.close()