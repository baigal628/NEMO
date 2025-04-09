# NEMO: a NEural network model for mapping MOdifications in nanopore Long-read  
<p align="left">
<img src="./img/nemo_logo.png" width="200"/>
</p>

# 🚀 Overview
NEMO is a deep learning tool designed to predict DNA modifications using nanopore long-read chromatin accessibility data. It allows users to train neural network models, predict modifications, and visualize results.

# ⚡ Installation

1. Clone the repository

```{bash}
git clone https://github.com/baigal628/NEMO.git
cd NEMO
```
2. Create and activate the conda environment:
```{bash}
conda create -n nemo python=3.9
conda activate nemo
```
3. Install dependencies:
```{bash}
pip install -r requirements.txt
```

# 🛠️ Basic utilities

## Navigate through test scripts to get full experience of NEMO functionalities
```{bash}
cd ./NEMO/test/sh/
```
Run the following bash scripts in their order
```{bash}
ll ./NEMO/test/sh/
0_pod5Tokmer.sh
1_train_r10.sh
2_predict_r10.sh
3_plot.sh
```

## 🔄 Data preprocessing for nanopore sequencing data

Basecall data using dorado: https://github.com/nanoporetech/dorado

```{bash}
# you do not need to run basecalling step for test dataset as we already provided basecalled bam file in the test data folder.

dorado basecaller dna_r9.4.1_e8_sup@v3.6 \
    ../input/test.pod5 \
    --emit-moves \
    --device cuda:all \
    --reference ../input/sacCer3.fa > ../input/test_reads.bam
```

Signal-to-Event Alignment: https://github.com/cafelton/pod5-to-kmer-signal

```{bash}
cat 0_pod5Tokmer.sh

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

```

## 📈 Train and test model using positive and negative control data
Preprocessed negative and positive data ar provided under ./nemo/test/output/

```{bash}
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
```
## 🔍 Predict modifications on chromatin data
```{bash}
python3  ../../src/predict.py \
    --bam ../input/test_reads.bam \
    --parquet ../output/test.parquet \
    --region 'chrI:500-2500' \
    --seq_len 400 \
    --step 200 \
    --weight ../output/test_r10_resnet_best_model.pt \
    --thread 4 \
    --outpath ../output/ \
    --prefix mod_prediction \
    --batch_size 216 \
```
## 📊 plot metagene at TSS

```{bash}
python3  ../../src/plot.py \
    --plot aggregate \
    --pred ../output/mod_prediction.tsv \
    --bed TSS.bed \
    --ref ../input/sacCer3.fa \
    --label 6mA \
    --outpath ../output/ \
    --prefix mod_prediction
```
# Reference

**Probing chromatin accessibility with small molecule DNA intercalation and nanopore sequencing**

Gali Bai*, Namrita Dhillon*, Colette Felton*, Brett Meissner*, Brandon Saint-John*, Robert Shelansky*, Elliot Meyerson, Eva Hrabeta-Robinson, Babak Hodjat, Hinrich Boeger, Angela N. Brooks
bioRxiv 2024.03.20.585815; doi: https://doi.org/10.1101/2024.03.20.585815

# 📬 Feedback & Contributions

We welcome contributions! Feel free to submit issues or pull requests to improve NEMO.

# ✨ Acknowledgments
Developed with ❤️ by Brooks Lab and Cognizant AI Labs. Thanks to the contributors and open-source community for their support!