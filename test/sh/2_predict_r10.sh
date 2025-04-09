python3  ../../src/predict.py \
    --bam ../input/chrom_ang_500.sorted.bam \
    --ref ../input/sacCer3.fa \
    --parquet ../output/chrom_ang_500-sigalign.parquet \
    --region all \
    --seq_len 400 \
    --step 200 \
    --weight ../output/ang_test_r10_resnet_best_model.pt \
    --thread 4 \
    --outpath ../output/ \
    --prefix ang_test_r10 \
    --batch_size 256

