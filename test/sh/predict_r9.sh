python3  ../src/predict.py \
    --sigalign pos_sig.tsv \
    --seqlen 40 \
    --step 20 \
    --weight /private/groups/brookslab/gabai/projects/Add-seq/data/train/240510_train_seqlen40/240510_train_addseq_seqlen40_resnet_best_model.pt \
    --thread 16 \
    --outpath ./output/pred/ \
    --prefix pos_sig \
    --batch_size 512 \


python3  ../src/predict.py \
    --sigalign neg_sig.tsv \
    --seqlen 40 \
    --step 20 \
    --weight /private/groups/brookslab/gabai/projects/Add-seq/data/train/240510_train_seqlen40/240510_train_addseq_seqlen40_resnet_best_model.pt \
    --thread 16 \
    --outpath ./output/pred/ \
    --prefix neg_sig \
    --batch_size 512 \

