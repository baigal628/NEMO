import argparse
import time
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from resnet1d import ResNet1D
from nanopore_convnet import NanoporeConvNet
from nanopore_dataset import create_sample_map
from nanopore_dataset import create_splits
from nanopore_dataset import load_sigalign
from nanopore_dataset import load_parquet
from nanopore_dataset import NanoporeDataset
from nanopore_transformer import NanoporeTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type= str, default='test')
parser.add_argument('--device', type= str, default='auto')

# input output parameters
parser.add_argument('--test_dataset', type= str, default='', help='test dataset')
parser.add_argument('--pred_out', type= str, default='', help='saved prediction output file in pickle format.')
parser.add_argument('--outpath', type=str, default='./')
parser.add_argument('--max_seq', type=int, default=0)

# model parameters
parser.add_argument('--model_type', default='resnet')
parser.add_argument('--weight', type = str, default='')
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()


##############
# set device #
##############
if args.device == 'auto':
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = args.device
print('Device type:', device)


################
# set up model #
################
assert args.model_type in ['convnet', 'resnet', 'transformer', 'phys']

if args.model_type == 'convnet':
    model = NanoporeConvNet().to(device)
elif args.model_type == 'resnet':
    model = ResNet1D(
                in_channels=1,
                base_filters=128,
                kernel_size=3,
                stride=2,
                groups=1,
                n_block=8,
                n_classes=2,
                downsample_gap=2,
                increasefilter_gap=4,
                use_do=False).to(device)
elif args.model_type == 'transformer':
    model = NanoporeTransformer(d_model=128,
                                dim_feedforward=256,
                                n_layers=6,
                                n_head=8).to(device)
elif args.model_type == 'phys':
    model = ResNet1D(
                in_channels=1,
                base_filters=128,
                kernel_size=16,
                stride=2,
                groups=32,
                n_block=48,
                n_classes=2,
                downsample_gap=6,
                increasefilter_gap=12,
                use_do=False).to(device)

model.load_state_dict(torch.load(args.weight, map_location=torch.device(device)))
model.to(device)
model.eval()
print("Created model and moved to the device.")


#######################
# create test dataset #
#######################

print("Preparing test dataset...")
start_time = time.time()
test_dataset = torch.load(args.test_dataset)
test_dataloader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
print("Prepared in  %s seconds." % (time.time() - start_time))

###############################
# show some performance stats #
###############################

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12


###########################
# predict on test dataset #
###########################

if not args.pred_out:
    print("Predicting on test dataset...")
    with torch.no_grad():
        seq_preds = {}
        dataloader_idx = 0
        for samples, labels in tqdm(test_dataloader):
            samples.to(device)
            pred = model(samples).sigmoid()
            for i in range(len(pred)):
                seq_idx = test_dataset.get_seq_idx(dataloader_idx)
                dataloader_idx += 1
                seq_label = labels[i].item()
                prediction = pred[i].item()
                seq_id = (seq_label, seq_idx)
                if seq_id not in seq_preds:
                    seq_preds[seq_id] = []
                seq_preds[seq_id].append(pred[i].item())
            if args.max_seq:
                if dataloader_idx >= args.max_seq:
                    break
    
    pred_out = f'{args.outpath}/{args.exp_id}_{args.model_type}_test_pred.pkl'

    # Save the Python object to the file
    with open(pred_out, 'wb') as file:
        pickle.dump(seq_preds, file)
    print(f"Python object saved to {pred_out}")

else:
    with open(args.pred_out, 'rb') as pred_outf:
        seq_preds = pickle.load(pred_outf)

    print("Loaded prediction data:", args.pred_out)



print('computing accuracy...')
correct = {0: 0, 1: 0}
total = {0: 0, 1: 0}

for seq_id in tqdm(seq_preds):
    label = seq_id[0]
    pred_arr = np.round(np.array(seq_preds[seq_id]))
    if label == 0:
        label_arr = np.zeros(len(pred_arr))
    else:
        label_arr = np.ones(len(pred_arr))
    correct_arr = (pred_arr == label_arr)
    correct[label] += np.sum(correct_arr)
    total[label] += len(pred_arr)

print(correct)
print(total)

accuracy = (correct[0] + correct[1]) / float(total[0] + total[1])  

true_negatives = correct[0]
true_positives = correct[1]
false_negatives = total[0] - correct[0]
false_positives = total[1] - correct[1]

tpr = true_positives/float(true_positives + false_negatives)
fpr = false_positives/float(true_negatives + false_positives)
tnr = true_negatives/float(true_negatives + false_positives)
fnr = false_negatives/float(true_positives + false_negatives)


accuracy = (correct[0] + correct[1]) / float(total[0] + total[1])
precision = true_positives / float(true_positives + false_positives)
recall = true_positives / float(true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)


print("True negatives:", true_negatives)
print("True positives:", true_positives)
print("False negatives:", false_negatives)
print("False positives:", false_positives)

print(f'tpr={tpr}, fpr={fpr}, tnr={tnr}, fnr={fnr}')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

performance_out = open(f'{args.outpath}/{args.exp_id}_{args.model_type}_test_pred.tsv', 'w')
performance_out.write(f'tpr\tfpr\ttnr\tfnr\taccuracy\tprecision\trecall\tf1\n')
performance_out.write(f'{tpr}\t{fpr}\t{tnr}\t{fnr}\t{accuracy}\t{precision}\t{recall}\t{f1}\n')
performance_out.close()

# Plot prediction mean and std for each sequence in test dateset
seq_means = {0: [], 1: []}
seq_stds = {0: [], 1: []}
for seq_id in tqdm(seq_preds):
    label = seq_id[0]
    seq_means[label].append(np.mean(seq_preds[seq_id]))
    seq_stds[label].append(np.std(seq_preds[seq_id]))
fig = plt.figure(figsize=(6,5))
plt.scatter(seq_means[0], seq_stds[0], label='negative control')
plt.scatter(seq_means[1], seq_stds[1], label='positive control')
plt.legend()
plt.xlabel('prediction mean')
plt.ylabel('prediction std')
plt.show()
plt.savefig(f'{args.outpath}/{args.exp_id}_{args.model_type}_test_mean_std.pdf', dpi = 1000)
plt.close()


seq_lens = {0: [], 1: []}
seq_accs = {0: [], 1: []}
for seq_id in tqdm(seq_preds):
    seq_len = len(seq_preds[seq_id])
    label = seq_id[0]
    if label == 0:
        label_arr = np.zeros(seq_len)
    else:
        label_arr = np.ones(seq_len)
    pred_arr = np.round(np.array(seq_preds[seq_id]))
    correct_arr = (pred_arr == label_arr)
    correct = np.sum(correct_arr)
    seq_acc = float(correct) / seq_len
    seq_lens[label].append(seq_len)
    seq_accs[label].append(seq_acc)
fig = plt.figure(figsize=(6,5))
plt.scatter(seq_lens[0], seq_accs[0], label='negative control')
plt.scatter(seq_lens[1], seq_accs[1], label='positive control')
plt.legend()
plt.xlabel('sequence length')
plt.ylabel('sequence accuracy')
plt.savefig(f'{args.outpath}/{args.exp_id}_{args.model_type}_test_accurracy_vs_seqlen.pdf', dpi = 1000)
plt.close()

# Compute ROC curve
print('computing roc...')
pred_list = []
label_list = []
for seq_id in tqdm(seq_preds):
    seq_len = len(seq_preds[seq_id])
    label = seq_id[0]
    preds = seq_preds[seq_id]
    if label == 0:
        labels = np.zeros(seq_len)
    else:
        labels = np.ones(seq_len)
    pred_list.append(preds)
    label_list.append(labels)
    
pred_cat = np.concatenate(pred_list)
label_cat = np.concatenate(label_list)

print('pos dataset:', list(label_cat).count(1))
print('neg dataset:', list(label_cat).count(0))

fpr, tpr, thresholds = roc_curve(label_cat, pred_cat)
roc_auc = auc(fpr, tpr)
fig = plt.figure(figsize=(6,5))
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.plot(fpr, tpr, color="darkorange", lw=2,
        label="AUC = %0.2f" % roc_auc)
plt.title("Receiver operating characteristic (ROC)", size = 'medium')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.savefig(f'{args.outpath}/{args.exp_id}_{args.model_type}_test_roc.pdf', dpi = 1000)
plt.close()
print('best cutoff:', thresholds[np.argmax(tpr - fpr)])

# Calculate kernel density estimate
pos_reads = []
neg_reads = []
for seq_id in tqdm(seq_preds):
    label = seq_id[0]
    if label == 0:
        neg_reads += seq_preds[seq_id]
    else:
        pos_reads += seq_preds[seq_id]
pos_kde = gaussian_kde(pos_reads)
neg_kde = gaussian_kde(neg_reads)
# Create a range of values for x-axis
pos_values = np.linspace(-0.01,1.01, 100)
neg_values = np.linspace(-0.01,1.01, 100)
# Plot the density curve
fig = plt.figure(figsize=(6,5))
plt.plot(pos_values, pos_kde(pos_values), label='positive control')
plt.plot(neg_values, neg_kde(neg_values), label='negative control')
# Add labels and title
plt.xlabel('Predicted scores')
plt.ylabel('Density')
plt.title('Density of predicted scores')
# Show legend
plt.legend()
# Show the plot
plt.savefig(f'{args.outpath}/{args.exp_id}_{args.model_type}_test_density.pdf', dpi = 1000)
plt.close()