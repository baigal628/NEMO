python3  ../src/ref/bampod5kmersig-witharrow-sigalign.py -b can_mappings.bam -p can_reads.pod5 -o can
python3 ../src/ref/bampod5kmersig-witharrow-sigalign.py -b mod_mappings.bam -p mod_reads.pod5 -o mod

python3  ../src/train_nemo_r10.py \
    --exp_id test_r10 \
    --neg_data can-sigalign.parquet \
    --pos_data mod-sigalign.parquet \
    --outpath ./output/ \
    --epochs 5 \
    --steps_per_epoch 50 \
    --val_steps_per_epoch 50 

python3  ~/tools/NEMO/src/test_model.py \
    --exp_id test_r10 \
    --test_dataset ./output/test_dataset_test_r10_resnet.pt \
    --weight ./test_r10_resnet_best_model.pt \
    --outpath  ./output/