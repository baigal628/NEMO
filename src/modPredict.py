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

def modPredict(bam, event, region, genome, prefix = '', modification = 'addseq', modbase = 'A', model = 'resnet1D', sigAlign = '', method = 'median',
               outPath = '/private/groups/brookslab/gabai/projects/Add-seq/data/ctrl/predictMod/', mbase = 'A',
               n_rname = 0):
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
    'HMR': 'chrIII:290000-299000',
    'AUA1': 'chrVI:114000-116000',
    'EMW1': 'chrXIV:45000-50000',
    'NRG2': 'chrII:370000-379000',
    'RDN37': 'chrXII:450300-459300'}
    
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
    
    print('Predicting on region: ', myregion)
    reg = myregion.split(':')
    chrom, pStart, pEnd = reg[0], int(reg[1].split('-')[0]), int(reg[1].split('-')[1])
    
    initial_time = time.time()
    print('Start parsing bam file to get reads aligned to the region......')
    alignment = getAlignedReads(sam = bam, region = myregion, genome=genome, print_name=False)
    refSeq = alignment['ref']
    modPositions = basePos(refSeq, base = modbase)
    rname = list(alignment.keys())
    print('done with bam file.')
    
    if not sigAlign:
        print('Start parsing eventalign file to map signals to the list......')
        sigAlign_output = outPath + prefix + '_' + myregion + 'siganlAlign.tsv'
        parseEventAlign(eventAlign = event, readname = rname[1:], chr_region=chrom, n_rname = n_rname,
                    outfile = sigAlign_output)
        print('done with reading eventalign file.')
        print('Output signalAlign file saved in:\n', sigAlign_output)
    else:
        print('Using user provided sigAlign file: ', sigAlign)
        sigAlign_output = sigAlign
    
    print('Start parsing sigalign file......')
    modScore_output = outPath + prefix + '_' + myregion + 'modScores.tsv'
    modScore_outF = open(modScore_output, 'w')
    
    # Relative position to pStart on reference genome:
    modScore_outF.write('#Positions\t' + ','.join(str(i) for i in modPositions)+ '\n')
    
    for readID, eventStart, sigList, siglenList in parseSigAlign(sigAlign=sigAlign_output):
        start_time = time.time()
        print('Start processing ', readID)
        strand = alignment[readID][1]
        sigLenList_init = pStart-eventStart-1
        if sigLenList_init > len(siglenList):
            continue
        scores = assign_scores(strand=strand, refSeq = refSeq, modPositions=modPositions, sigList=sigList, siglenList=siglenList, sigLenList_init=sigLenList_init,
                               weights = myweights, model= mymodel, device = device, tune=False, method = method)
        probs = ','.join(str(i) for i in scores)
        out = '{readID}\t{chrom}\t{strand}\t{pStart}\t{pEnd}\t{probs}\n'.format(readID=readID, chrom = chrom, strand = strand, pStart=pStart, pEnd=pEnd, probs=probs)
        modScore_outF.write(out)
        
        end_time = time.time()
        print('Processed read ', readID, ' in ', end_time - start_time, 's')
        print('Output signalAlign file saved in:\n', modScore_output)
    modScore_outF.close()
    total_time = time.time() - initial_time
    print('Finished all analysis in ', total_time, 's.')