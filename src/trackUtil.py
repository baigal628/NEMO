import numpy as np

def writeBedGraph(bedGraphHeader, count, coverage, bins, step, outfile, chrom, qstart, qend, track_name = '', track_color = '', normalize = True):
    
    thisheader = bedGraphHeader
    
    if track_name:
        thisheader['name'] = track_name
    if track_color:
        thisheader['color'] = track_color
    
    outFh = open(outfile, 'w')
    
    for k,v in bedGraphHeader.items():
        if v:
            line = k + '=' + v + ' '
            outFh.write(line)
    outFh.write('\n')

    for i in range(len(bins)):
        start, end = bins[i], bins[i]+step
        if coverage[i] == 0:
            binHeight = 0
        elif normalize:
            binHeight = "%.3f" % (count[i]/coverage[i])
        else:
            binHeight = count[i]
        line = '{chr}\t{start}\t{end}\t{binheight}\n'.format(chr = chrom, start = start,  end = end, binheight = binHeight)
        outFh.write(line)
    outFh.close()

def predictionToBedGraph(prediction, bins, step, threshold, chrom, qstart, qend, prefix, outfile):

    count, cov = np.zeros(len(bins), dtype = int), np.zeros(len(bins), dtype = int)
    
    bedGraphHeader = {'track type':'bedGraph', 
                  'name':'Nemo prediction', 
                  'description':'addseq',
                  'visibility':'', 
                  'color':'r', 
                  'altColor':'r', 
                  'priority':'', 
                  'autoScale':'off', 
                  'alwaysZero':'off', 
                  'gridDefault':'off', 
                  'maxHeightPixels':'default', 
                  'graphType':'bar',
                  'viewLimits':'upper',
                  'yLineMark':'',
                  'yLineOnOff':'on',
                  'windowingFunction':'mean',
                  'smoothingWindow':'on'
                 }

    
    with open(prediction, 'r') as predFh:
        for line in predFh:
            line = line.strip().split('\t')
            readname = line[0]
            strand = line[1]
            binStart = int(line[2])
            if binStart > bins[-1]:
                continue
            probs = line[3].split(',')
            binEnd = binStart + step*(len(probs)-1)
            if binEnd < bins[0]:
                continue
            else:
                i = int((binStart-bins[0])/step)
                if i < 0:
                    probs = probs[-i:]
                    i = 0
                for prob in probs:
                    if float(prob)>= float(threshold):
                        count[i]+=1
                    cov[i] +=1
                    i+=1
                    if i >= len(bins):
                        break
    writeBedGraph(bedGraphHeader=bedGraphHeader, count=count , coverage=cov, bins=bins, step=step, 
                  chrom= chrom, qstart=qstart, qend=qend, outfile=outfile, track_name=prefix, track_color = 'blue')