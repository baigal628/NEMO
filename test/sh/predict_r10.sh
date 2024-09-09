# python3  ../../src/predict.py \
#     --bam /private/groups/brookslab/gabai/projects/Add-seq/data/chrom/sphe/220524_500Ang_Spheroblast_dorado_movesOut_sorted.bam \
#     --parquet /private/groups/brookslab/gabai/projects/Add-seq/data/chrom/sphe/240808_sphe-sigalign.parquet \
#     --region 'chrXVI:66400-67550' \
#     --seq_len 400 \
#     --step 200 \
#     --weight /private/groups/brookslab/gabai/projects/Add-seq/data/train/240822_train_addseq/240827_train_addseq_seqlen400_50_seq_per_batch_batch_128_epoch100_resnet_best_model.pt \
#     --thread 16 \
#     --outpath /private/groups/brookslab/gabai/projects/Add-seq/data/chrom/modPredict/240828_prediction/ \
#     --prefix 240827_dorado_sphe_pos_seqlen400_50_seq_per_batch_batch_128_CLN2 \
#     --batch_size 216 \
#     --mean 77.0 \
#     --std 20.0 \

# python3  ../../src/predict.py \
#     --bam /private/groups/brookslab/gabai/projects/Add-seq/data/chrom/sphe/220524_500Ang_Spheroblast_dorado_movesOut_sorted.bam \
#     --parquet /private/groups/brookslab/gabai/projects/Add-seq/data/chrom/sphe/240808_sphe-sigalign.parquet \
#     --ref /private/groups/brookslab/gabai/tools/ref/yst/sacCer3.fa \
#     --seq_len 400 \
#     --step 200 \
#     --weight /private/groups/brookslab/gabai/projects/Add-seq/data/train/240822_train_addseq/240827_train_addseq_seqlen400_50_seq_per_batch_step1_batch_256_epoch100_resnet_best_model.pt  \
#     --thread 16 \
#     --outpath /private/groups/brookslab/gabai/projects/Add-seq/data/chrom/modPredict/240828_prediction/ \
#     --prefix 240827_dorado_sphe_pos_seqlen400_50_seq_per_batch_batch_256 \
#     --batch_size 256 \


python3  ../../src/predict.py \
    --bam /private/groups/brookslab/gabai/projects/Add-seq/data/chrom/sphe/220524_500Ang_Spheroblast_dorado_movesOut_sorted.bam \
    --parquet /private/groups/brookslab/gabai/projects/Add-seq/data/chrom/sphe/240808_sphe-sigalign.parquet \
    --ref /private/groups/brookslab/gabai/tools/ref/yst/sacCer3.fa \
    --seq_len 400 \
    --step 200 \
    --weight /private/groups/brookslab/gabai/projects/Add-seq/data/train/240822_train_addseq/240827_train_addseq_seqlen400_50_seq_per_batch_step1_batch_256_epoch100_resnet_best_model.pt  \
    --thread 16 \
    --mean 77.0 \
    --std 20.0 \
    --outpath /private/groups/brookslab/gabai/projects/Add-seq/data/chrom/modPredict/240828_prediction/ \
    --prefix 240827_dorado_sphe_pos_seqlen400_50_seq_per_batch_batch_256_normalized \
    --batch_size 256 \