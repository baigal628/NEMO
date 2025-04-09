python3 ../../src/ref/bampod5kmersig-witharrow-sigalign.py \
    -b ../input/ang_0.sorted.bam \
    -p ../input/ang_0_downsampled.pod5 \
    -o ../output/ang_0

python3 ../../src/ref/bampod5kmersig-witharrow-sigalign.py \
    -b ../input/ang_500.sorted.bam \
    -p ../input/ang_500_downsampled.pod5 \
    -o ../output/ang_500

python3 ../../src/ref/bampod5kmersig-witharrow-sigalign.py \
    -b ../input/chrom_ang_500.sorted.bam \
    -p ../input/chrom_ang_500_downsampled.pod5 \
    -o ../output/chrom_ang_500