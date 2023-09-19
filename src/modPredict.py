from seqUtil import *
from bamUtil import *
from nanoUtil import *
from nntUtil import *

def modPredict(bam, event, region, genome = ref, prefix = '', modification = 'addseq', model = 'resnet1D',
               outPath = '/private/groups/brookslab/gabai/projects/Add-seq/data/ctrl/eventalign/'):
    '''
    Given bam file, sigAlign file, region, and s
    '''
    nuc_regions = {
    'PHO5': 'chrII:429000-435000',
    'CLN2': 'chrXVI:66000-67550',
    'HMR': 'chrIII:290000-299000'}
    
    if region in nuc_regions:
        myregion = nuc_regions[region]
    else:
        myregion = region
    
    models = {
        'resnet1D':resnet1D
    }
    
    weights = {
        'addseq':ADDSEQ_FN,
        'mesmelseq':MESMLR_FN
    }

    mymodel = models[model]
    myweights = weights[modification]
    
    reg = myregion.split(':')
    chrom, pStart, pEnd = reg[0], int(reg[1].split('-')[0]), int(reg[1].split('-')[1])
    
    print('Start parsing bam file to get reads aligned to the region......')
    alignment = getAlignedReads(sam = bam, region = myregion, genome=genome, print_name=False)
    rname = list(alignment.keys())
    rstrand = {r:s[1] for r, s in alignment.items()}
    ref_seq = alignment['ref']
    print('done with bam file.')
    
    print('Start parsing eventalign file to map signals to the list......')
    output_file = outPath + prefix + '_' + myregion + 'siganlAlign.tsv'
    parseEventAlign(eventAlign = event, readname = rname[1:], genome=genome, chrom=chrom, n_rname = 3,
                outfile = output_file)
    print('done with reading eventalign file.')
    print('Output signalAlign file saved in:\n', output_file)
    
    print('Start parsing sigalign file......')
    for readID, sigList, siglenList, sigStart in parseSigAlign(sigAlign=output_file, pStart=pStart, pEnd=pEnd):
        print('Making modification predictions on read: ', readID)
        if rstrand[readID] == -1: ref_seq = reverseCompliment(ref_seq)
        output_file = outPath + prefix + '_' + readID + '_' + myregion + 'modelScores.tsv'
        modelScores(refSeq = ref_seq, sigList = sigList, siglenList = siglenList, sigStart=sigStart, outfile=output_file,
                    device = 'cpu', model = mymodel, weights_path = myweights)