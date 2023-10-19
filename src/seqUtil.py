import pysam
import numpy
import random
# from IPython.display import display, HTML

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

def randomPosition(n, genome, windowSize, mode = 'even',):
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
    ntDict = {'A': 'T', 'C': 'G', 'G': 'C', 'T':'A', 'D':'D', 'N':'N'}
    return ''.join([ntDict[i] for i in seq[::-1]])

def compliment(seq):
    '''get compliment of sequence given input seq.'''
    ntDict = {'A': 'T', 'C': 'G', 'G': 'C', 'T':'A', 'D':'D', 'N':'N'}
    return ''.join([ntDict[i] for i in seq])

def format_chars(seq):
    '''
    Color print nt sequences in interactive mode.
    Usage:
        format_chars{'ACGTDN'}
    
    '''
    colorMap = {'A':'0.7', 'C': '-0.7', 'G': '-0.5', 'T': '-1.2', "D": '0.3',"N": '2.0'}
    chars = [i for i in seq]
    numbers = [colorMap[i] for i in seq]
    
    numbers = np.array(numbers).astype(float)
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = cm.RdYlGn
    colors = cmap(norm(numbers))
    hexcolor = [mcolors.to_hex(c) for c in colors]
    letter = lambda x: "<span style='color:{};'>{}</span>".format(x[1],x[0])
    text = "".join(list(map(letter, zip(chars,hexcolor))))
    text = "<div style='font-size:8pt;'>" + text
    display(HTML(text))

def getchromOrder(genome):
    '''
    Given a genome, get the chromosome order dictionary
        E.g. for yeast genome, you get{'chrI': 0,
                                     'chrII': 1,
                                     'chrIII': 2,
                                     'chrIV': 3,
                                     'chrIX': 4,
                                     'chrV': 5,
                                     'chrVI': 6,
                                     'chrVII': 7,
                                     'chrVIII': 8,
                                     'chrX': 9,
                                     'chrXI': 10,
                                     'chrXII': 11,
                                     'chrXIII': 12,
                                     'chrXIV': 13,
                                     'chrXV': 14,
                                     'chrXVI': 15,
                                     'chrM': 16}
    '''
    
    genome_size = fetchSize(genome)
    chromOrder = {j:i for i, j in enumerate(genome_size.keys())}
    return chromOrder

def baseCount(seq, base):
    count = 0
    for nt in seq:
        if nt == base:
            count+=1
    return count

def basePos(seq, base):
    pos = []
    i = 0
    for nt in seq:
        if nt == base:
            pos.append(i)
        i +=1
    return pos