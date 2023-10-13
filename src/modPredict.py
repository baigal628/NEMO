from seqUtil import *
from bamUtil import *
from nanoUtil import *
from nntUtil import *
import time


ntDict = {'A': 'T', 'C': 'G', 'G': 'C', 'T':'A', 'D':'D', 'N':'N'}

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print('Device type: ', device)
ADDSEQ_FN = '/private/groups/brookslab/gabai/tools/seqUtils/src/nanopore_classification/best_models/addseq_resnet1d.pt'
MESMLR_FN = '/private/groups/brookslab/gabai/tools/seqUtils/src/nanopore_classification/best_models/mesmlr_resnet1d.pt'

def modPredict(bam, event, region, genome, prefix = '', modification = 'addseq', model = 'resnet1D', sigAlign = '', method = 'median',
               outPath = '/private/groups/brookslab/gabai/projects/Add-seq/data/ctrl/eventalign/predictMod', mbase = 'A',
               n_rname = 3):
    '''
    Given bam file and event align file, do modification prediction on all modifiable bases.
    Input:
        bam: sorted, indexed, and binarized alignment file.
        event: evenalign file from nanopolish eventalign output.
        region: fenomic coordinates to perform modification predictions. 
                Format: chrI:1-1000 or any of predifined regions: 'PHO5', 'CLN2', 'HMR'.
        genome: path to the reference genome.fa file.
        modification: modification types, one of: 'addseq', 'mesmelseq'.
        model = deep learning model for predicting modifications.
        mbase: modifiable base. One of: 'A', 'C', 'G', 'T'.
        n_rname: number of reads to be skipped for reads mapped to the defined region.

    Output:
        prefix: prefix attached to the output file names.
        outPath: path to store the output files.
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
    print('done with bam file.')
    
    if not sigAlign:
        print('Start parsing eventalign file to map signals to the list......')
        sigAlign_output = outPath + prefix + '_' + myregion + 'siganlAlign.tsv'
        parseEventAlign(eventAlign = event, readname = rname[1:], genome=genome, chrom=chrom, n_rname = n_rname,
                    outfile = sigAlign_output)
        print('done with reading eventalign file.')
        print('Output signalAlign file saved in:\n', sigAlign_output)
    else:
        print('Using user provided sigAlign file: ', sigAlign)
        sigAlign_output = sigAlign
    
    print('Start parsing sigalign file......')
    modScore_output = outPath + prefix + '_' + myregion + 'modScores.tsv'
    modScore_outF = open(modScore_output, 'w')
    for readID, sigList, siglenList, sigStart in parseSigAlign(sigAlign=sigAlign_output, pStart=pStart, pEnd=pEnd):
        start_time = time.time()
        print('Start processing ', readID)
        
        modScores = assign_scores(readID=readID, sigList=sigList, siglenList=siglenList, sigStart=sigStart, modbase= 'A', 
                                  alignemnt=alignment, model= mymodel, weights = myweights, device = device,tune=False, method = method)
        
        out = '{readID}\t{chrom}\t{pStart}\t{pEnd}\t{prob}'.format(readID=readID, chrom = chrom, pStart=pStart, pEnd=pEnd, prob=str(modScores))
        modScore_outF.write(out)
        
        end_time = time.time()
        print('Processed read ', readID, ' in ', end_time - start_time, 's')
        print('Output signalAlign file saved in:\n', modScore_output)
    modScore_outF.close()