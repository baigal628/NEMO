"""
Entrypoint for training nanopore classification models
"""

import argparse
import pandas as pd
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
from nanopore_dataset import load_csv
from nanopore_dataset import NanoporeDataset
from nanopore_transformer import NanoporeTransformer


# Read commandline arguments

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', default='test')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--neg_data', default='data/mesmlr/reprocessed-neg.eventalign.signal.csv')
parser.add_argument('--pos_data', default='data/mesmlr/reprocessed-pos.eventalign.signal.csv')
parser.add_argument('--seq_len', type=int, default=400)
parser.add_argument('--min_val', type=float, default=50) # Used to clip outliers
parser.add_argument('--max_val', type=float, default=130) # Used to clip outliers
parser.add_argument('--train_split', type=float, default=0.8)
parser.add_argument('--val_split', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--steps_per_epoch', type=int, default=1000)
parser.add_argument('--val_steps_per_epoch', type=int, default=1000)
parser.add_argument('--model_type', default='resnet')
parser.add_argument('--opt', default='adam')
parser.add_argument('--decay', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--max_seqs', type=int, default=None)
args = parser.parse_args()


# Set results file names
best_model_fn = f'best_models/{args.exp_id}.pt'
metrics_fn = f'results/{args.exp_id}.csv'

# Prepare data for training
print("Preparing unmodified...")
print("Loading csv...")
unmodified_sequences = load_csv(args.neg_data,
                                min_val=args.min_val,
                                max_val=args.max_val,
                                max_sequences=args.max_seqs)
print("Creating sample map...")
unmodified_sample_map = create_sample_map(unmodified_sequences,
                                          seq_len=args.seq_len)

print("Creating splits...")
unmodified_train, unmodified_val, unmodified_test = create_splits(
        unmodified_sequences, unmodified_sample_map, seq_len=args.seq_len, shuffle=False)
print("Prepared.")

print("Preparing modified...")
print("Loading csv...")
modified_sequences = load_csv(args.pos_data,
                              min_val=args.min_val,
                              max_val=args.max_val,
                              max_sequences=args.max_seqs)
print("Creating sample map...")
modified_sample_map = create_sample_map(modified_sequences,
                                        seq_len=args.seq_len)
print("Creating splits...")
modified_train, modified_val, modified_test = create_splits(
        modified_sequences, modified_sample_map, seq_len=args.seq_len, shuffle=False)
print("Prepared.")

train_dataset = NanoporeDataset(unmodified_sequences,
                                unmodified_train,
                                modified_sequences,
                                modified_train,
                                device=args.device,
                                synthetic=False,
                                seq_len=args.seq_len)

val_dataset = NanoporeDataset(unmodified_sequences,
                              unmodified_val,
                              modified_sequences,
                              modified_val,
                              device=args.device,
                              synthetic=False,
                              seq_len=args.seq_len)

test_dataset = NanoporeDataset(unmodified_sequences,
                               unmodified_test,
                               modified_sequences,
                               modified_test,
                               device=args.device,
                               synthetic=False,
                               seq_len=args.seq_len)

train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
val_dataloader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)


# Create model

assert args.model_type in ['convnet', 'resnet', 'transformer', 'phys']
if args.model_type == 'convnet':
    model = NanoporeConvNet(input_size=args.seq_len).to(args.device)
    summary(model, (1, 400))
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
                use_do=False).to(args.device)
    summary(model, (1, 400))
elif args.model_type == 'transformer':
    model = NanoporeTransformer(d_model=128,
                                dim_feedforward=256,
                                n_layers=6,
                                n_head=8).to(args.device)
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
                use_do=False).to(args.device)
    summary(model, (1, 400))

print("Created model and moved to device")


# Create loss function and optimizer

loss_function = nn.BCEWithLogitsLoss()

assert args.opt in ['adam', 'sgd']
if args.opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
elif args.opt == 'sgd':
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          weight_decay=args.decay,
                          momentum=args.momentum,
                          nesterov=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='max',
                                                 factor=0.1,
                                                 patience=args.patience)


# Train model
best_val_acc = -1
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(args.epochs):

    model.train()

    # Set up progress bar
    print()
    progress = tqdm(range(args.steps_per_epoch),
                    desc=f'Epoch {epoch+1}: Training')

    total_loss = 0.0
    correct = 0
    total = 0
    for step, (sample, label) in enumerate(train_dataloader):

        # Run batch
        sample.to(args.device)
        label.to(args.device)
        model.zero_grad()
        pred = model(sample)
        loss = loss_function(pred, label)
        loss.backward()
        optimizer.step()

        # Update running metrics
        pred_label = torch.round(torch.sigmoid(pred))
        total += label.size(0)
        correct += (pred_label == label).sum().item()
        total_loss += loss.item()
        train_loss = total_loss / float(total)
        train_acc = correct / float(total)

        # Update progress bar
        progress.set_postfix({
            'train_loss': train_loss,
            'train_acc': train_acc,
        })
        progress.update()

        if step == args.steps_per_epoch:
            break


    # Evaluate on validation set

    model.eval()
    with torch.no_grad():

        # Set up progress bar
        print()
        progress = tqdm(range(args.val_steps_per_epoch),
                        desc=f'Epoch {epoch+1} Validating')

        total_loss = 0.0
        correct = 0
        total = 0
        for step, (sample, label) in tqdm(enumerate(val_dataloader)):

            # Run batch
            sample.to(args.device)
            label.to(args.device)
            pred = model(sample)
            loss = loss_function(pred, label)

            # Update metrics
            total_loss += loss.item()
            pred_label = torch.round(torch.sigmoid(pred))
            total += label.size(0)
            correct += (pred_label == label).sum().item()
            val_loss = total_loss / float(total)
            val_acc = correct / float(total)

            # Update progress bar
            progress.set_postfix({
                'val_loss': val_loss,
                'val_acc': val_acc,
            })
            progress.update()

            # Update scheduler
            if step == args.val_steps_per_epoch:
                scheduler.step(val_acc)
                print(f'\nValidation accuracy: {val_acc} ')
                break

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'New best val acc: {best_val_acc}')
            torch.save(model.state_dict(), best_model_fn)
            print('Model saved.')

        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        metrics_df = pd.DataFrame({
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs})
        metrics_df.to_csv(metrics_fn)

