import pysam
import numpy
import random

def fetchSize(genome):
    '''Input a genome.fa file, get the chromsome bounderies.'''
    
    genomeSize = {}
    sequence = ''
    with open(genome, 'r') as refFh:
        for line in refFh:
            line = line.strip()
            if '>' in line:
                if sequence:
                    genomeSize[chrom] = len(sequence)
                    sequence = ''
                    chrom = line.split('>')[1]
                else:
                    chrom = line.split('>')[1]
            else:
                sequence += str(line)
        if sequence:
            genomeSize[chrom] = len(sequence)
    return genomeSize

def randomPosition(n, genome, windowSize, mode = 'even',):predictedThreshold.tsv
    ''' Input a genome.fa file, curate random genome positions with set length of windowSize'''
    
    genomeSize = fetchSize(genome)
    randomPos = {}
    for chrom, border in genomeSize.items():
        for i in range(n):
            startPos = random.randrange(0, border-windowSize+1)
            if chrom not in randomPos:
                randomPos[chrom] = [startPos]
            else:
                randomPos[chrom].append(startPos)
    return randomPos

def reverseCompliment(seq):
    '''get reverse compliment of sequence given input seq.'''
    ntDict = {'A': 'T', 'C': 'G', 'G': 'C', 'T':'A'}
    return ''.join([ntDict[i] for i in seq[::-1]])

def compliment(seq):
    '''get compliment of sequence given input seq.'''
    ntDict = {'A': 'T', 'C': 'G', 'G': 'C', 'T':'A'}
    return ''.join([ntDict[i] for i in seq])