# NEMO: a NEural network model for mapping MOdifications in nanopore Long-read 
<p align="left">
<img src="./md/nemo_logo.png" width="200"/>
</p>

# Utilities
```{python}
python3 findNemo.py --help

usage: findNemo.py [-h] [--mode MODE] [--region REGION] [--bam BAM] [--genome GENOME] [--eventalign EVENTALIGN] [--sigalign SIGALIGN]
                   [--outpath OUTPATH] [--prefix PREFIX] [--model MODEL] [--weight WEIGHT] [--threads THREADS] [--step STEP]
                   [--kmerWindow KMERWINDOW] [--signalWindow SIGNALWINDOW] [--load LOAD] [--prediction PREDICTION] [--gtf GTF]
                   [--pregion PREGION]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE, -mode MODE
                        two modes available: [predict, plot]
  --region REGION, -r REGION
                        genomic coordinates to perform modification predictions. E.g. chrI:2000-5000 or chrI.
  --bam BAM, -b BAM     sorted, indexed, and binarized alignment file.
  --genome GENOME, -g GENOME
                        reference genome fasta file
  --eventalign EVENTALIGN, -e EVENTALIGN
                        nanopolish eventalign file.
  --sigalign SIGALIGN, -s SIGALIGN
                        sigalign file if sigAlign file already exist. If not, must provide eventalign to generate sigAlign file.
  --outpath OUTPATH, -o OUTPATH
                        path to store the output files.
  --prefix PREFIX, -p PREFIX
                        prefix of output file names.
  --model MODEL, -m MODEL
                        deep neural network meodel used for prediction.
  --weight WEIGHT, -w WEIGHT
                        path to model weight.
  --threads THREADS, -t THREADS
                        number of threads.
  --step STEP, -step STEP
                        step to bin the region.
  --kmerWindow KMERWINDOW, -kw KMERWINDOW
                        kmer window size to extend bin.
  --signalWindow SIGNALWINDOW, -sw SIGNALWINDOW
                        signal Window size to feed into the model.
  --load LOAD, -l LOAD  number of reads to load into each iterations. Each iteration will output a file.
  --prediction PREDICTION, -pred PREDICTION
                        path to prediction file from modification prediction results.
  --gtf GTF, -gtf GTF   path to General Transfer Format (GTF) file.
  --pregion PREGION, -pregion PREGION
                        region to plot. Can be gene name of the pre defined gene regions.
```

# II. Example:
## 1. Preidct modified regions using pre-trained model

```{bash}
python3 ./findNemo.py
    --mode predict \
    --region chrII \
    --bam ./Add-seq/data/chrom/mapping/chrom.sorted.bam \
    --genome ./Add-seq/data/ref/sacCer3.fa \
    --eventalign ./addseq_data/eventalign/chrII.eventalign.txt \
    --outpath ./addseq_data/231110_test_nemo_v0_chrII/ \
    --prefix 231110_addseq_chrII \
    --threads 8

# With pre computed sigalign file:
python3 ./findNemo.py
    --mode predict \
    --region chrII \
    --bam ./Add-seq/data/chrom/mapping/chrom.sorted.bam \
    --genome ./Add-seq/data/ref/sacCer3.fa \
    --sigalign ./addseq_data/231110_test_nemo_v0_chrII/231110_test_chrIIchrII_sig.tsv \
    --outpath ./addseq_data/231110_test_nemo_v0_chrII/ \
    --prefix 231110_addseq_chrII \
    --threads 8
```

## 2. plot modified regions using prediction file
```{bash}
python3 ./findNemo.py
    --mode plot \
    --region chrII \
    --bam ./Add-seq/data/chrom/mapping/chrom.sorted.bam \
    --genome ./Add-seq/data/ref/sacCer3.fa \
    --outpath ./addseq_data/231110_test_nemo_v0_chrII/ \
    --prefix 231110_addseq_chrII \
    --prediction  ./231110_addseq_chrII_0_prediction.tsv
    --pregion PHO5
    --gtf ./data/ref/Saccharomyces_cerevisiae.R64-1-1.109.gtf
```

