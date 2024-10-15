python3  ../../src/predict.py \
    --bam ../input/mod_mappings.bam \
    --parquet ../output/mod-sigalign.parquet \
    --region 'chr13:52306200-52312800' \
    --seq_len 400 \
    --step 200 \
    --weight ../output/test_r10_resnet_best_model.pt \
    --thread 4 \
    --outpath ../output/ \
    --prefix mod_prediction \
    --batch_size 216 \

