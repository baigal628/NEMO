python3  ../../src/ref/bampod5kmersig-witharrow-sigalign.py -b ../input/can_mappings.bam -p ../input/can_reads.pod5 -o ../output/can
python3 ../../src/ref/bampod5kmersig-witharrow-sigalign.py -b ../input/mod_mappings.bam -p ../input/mod_reads.pod5 -o ../output/mod

python3 ../../src/train.py \
    --exp_id test_r10 \
    --neg_data ../output/can-sigalign.parquet \
    --pos_data ../output/mod-sigalign.parquet \
    --batch_size 256 \
    --seq_len 400 \
    --model_type resnet \
    --outpath ../output/ \
    --save_test \
    --epochs 5 \
    --steps_per_epoch 20 \
    --val_steps_per_epoch 10 

python3 ../../src/test.py \
    --exp_id test_r10 \
    --model_type resnet \
    --test_dataset ../output/test_dataset_test_r10_resnet.pt \
    --weight ../output/test_r10_resnet_best_model.pt \
    --outpath  ../output/ \
    --batch_size 256