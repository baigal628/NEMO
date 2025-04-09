python3 ../../src/train.py \
    --exp_id ang_test_r10 \
    --neg_data ../output/ang_0-sigalign.parquet \
    --pos_data ../output/ang_500-sigalign.parquet \
    --batch_size 256 \
    --seq_len 400 \
    --model_type resnet \
    --outpath ../output/ \
    --save_test \
    --epochs 5 \
    --steps_per_epoch 20 \
    --val_steps_per_epoch 10 

python3 ../../src/test.py \
    --exp_id ang_test_r10 \
    --model_type resnet \
    --test_dataset ../output/test_dataset_ang_test_r10_resnet.pt \
    --weight ../output/ang_test_r10_resnet_best_model.pt \
    --outpath  ../output/ \
    --batch_size 256