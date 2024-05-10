python3  ../src/train.py \
    --exp_id test_r9 \
    --neg_data neg_sig.tsv \
    --pos_data pos_sig.tsv \
    --outpath ./output/ \
    --epochs 5 \
    --steps_per_epoch 50 \
    --val_steps_per_epoch 50 \
    --save_test \
    --input_dtype sigalign

python3  ../src/src/test.py \
    --exp_id test_r9 \
    --test_dataset ./output/test_dataset_test_r9_resnet.pt \
    --weight ./output/test_r9_resnet_best_model.pt \
    --outpath  ./output/ \
    --pred_out ./output/test_r9_resnet_test_pred.pkl

python3   ../src/train_test_split.py \
    --exp_id test_train_test_split \
    --neg_data neg_sig.tsv \
    --pos_data pos_sig.tsv \
    --outpath ./output/ \
    --input_dtype sigalign