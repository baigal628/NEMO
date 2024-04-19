python3  ~/tools/NEMO/src/ref/bampod5kmersig-witharrow-sigalign.py -b can_mappings.bam -p can_reads.pod5 -o ../sigalign/can
python3 ~/tools/NEMO/src/ref/bampod5kmersig-witharrow-sigalign.py -b mod_mappings.bam -p mod_reads.pod5 -o ../sigalign/mod

python3  ~/tools/NEMO/src/train_nemo_r10.py \
    --exp_id test_r10_pod5_default \
    --neg_data ~/projects/ontMod/data/sigalign/can-sigalign.parquet \
    --pos_data ~/projects/ontMod/data/sigalign/mod-sigalign.parquet \
    --outpath ~/projects/ontMod/data/train/ \
    --epochs 5 \
    --steps_per_epoch 50 \
    --val_steps_per_epoch 50 

python3  ~/tools/NEMO/src/test_model.py \
    --exp_id test_r10_pod5 \
    --test_dataset ~/projects/ontMod/data/train/test_dataset_test_r10_pod5_default_resnet.pt \
    --weight ~/projects/ontMod/data/train/test_r10_pod5_default_resnet_best_model.pt \
    --outpath  ~/projects/ontMod/data/train/