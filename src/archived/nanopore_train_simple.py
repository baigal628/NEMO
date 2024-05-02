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
from nanopore_dataset import load_sigalign
from nanopore_dataset import NanoporeDataset
from nanopore_transformer import NanoporeTransformer

# Read commandline arguments

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', default='test')
parser.add_argument('--device', default='cuda:0')
# if you do not have a train validation split
parser.add_argument('--neg_data', default='')
parser.add_argument('--pos_data', default='')
parser.add_argument('--neg_train', default='')
parser.add_argument('--pos_train', default='')
parser.add_argument('--neg_val', default='')
parser.add_argument('--pos_val', default='')
parser.add_argument('--neg_seq', default='')
parser.add_argument('--pos_seq', default='')
parser.add_argument('--seq_len', type=int, default=400)
parser.add_argument('--min_val', type=float, default=50) # Used to clip outliers
parser.add_argument('--max_val', type=float, default=130) # Used to clip outliers
parser.add_argument('--train_split', type=float, default=0.8)
parser.add_argument('--val_split', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=128)
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
parser.add_argument('--outpath', type=str, default='')
args = parser.parse_args()

##############
# set device #
##############
if args.device == 'auto':
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = args.device
print('Device type:', device)

###############
# set outfile #
###############
# best model based on validation accuracy
best_model_fn = f'{args.outpath}/best_models/{args.exp_id}_{args.model_type}.pt'
# stats associated with best model
best_model_accuracy_fn = f'{args.outpath}/results/{args.exp_id}_{args.model_type}_best_model.csv'
# stats associated with best model
metrics_fn = f'{args.outpath}/results/{args.exp_id}_{args.model_type}.csv'

# unmodified input data
if args.neg_data:
    # Prepare data for training
    print("Preparing unmodified...")
    print("Loading csv...")
    unmodified_sequences = load_sigalign(args.neg_data,
                                        min_val=args.min_val,
                                        max_val=args.max_val,
                                        max_sequences=args.max_seqs)
    print("Creating sample map...")
    unmodified_sample_map = create_sample_map(unmodified_sequences,
                                            seq_len=args.seq_len)

    print("Creating splits...")
    unmodified_train, unmodified_val, unmodified_test = create_splits(
            unmodified_sequences, unmodified_sample_map, seq_len=args.seq_len, shuffle=True)
    print("Prepared.")
    del unmodified_sample_map

elif args.neg_train:
    print("Reading unmodified...")
    unmodified_sequences = torch.load(args.neg_seq)
    unmodified_train = torch.load(args.neg_train)
    unmodified_val = torch.load(args.neg_val)
else:
    print('No unmodified data provided!')

# modified input data
if args.pos_data:
    print("Preparing modified...")
    print("Loading csv...")
    modified_sequences = load_sigalign(args.pos_data,
                                    min_val=args.min_val,
                                    max_val=args.max_val,
                                    max_sequences=args.max_seqs)
    print("Creating sample map...")
    modified_sample_map = create_sample_map(modified_sequences,
                                            seq_len=args.seq_len)
    print("Creating splits...")
    modified_train, modified_val, modified_test = create_splits(
            modified_sequences, modified_sample_map, seq_len=args.seq_len, shuffle=True)
    print("Prepared.")

    del modified_sample_map

elif args.pos_train:
    print("Reading modified...")
    modified_sequences = torch.load(args.pos_seq)
    modified_train = torch.load(args.pos_train)
    modified_val = torch.load(args.pos_val)
else:
    print('No modified data provided!')

train_dataset = NanoporeDataset(unmodified_sequences,
                                unmodified_train,
                                modified_sequences,
                                modified_train,
                                device=device,
                                synthetic=False,
                                seq_len=args.seq_len)
del unmodified_train, modified_train

val_dataset = NanoporeDataset(unmodified_sequences,
                              unmodified_val,
                              modified_sequences,
                              modified_val,
                              device=device,
                              synthetic=False,
                              seq_len=args.seq_len)
del unmodified_val, modified_val

# test_dataset = NanoporeDataset(unmodified_sequences,
#                                unmodified_test,
#                                modified_sequences,
#                                modified_test,
#                                device=device,
#                                synthetic=False,
#                                seq_len=args.seq_len)
# del unmodified_test, modified_test
del unmodified_sequences, modified_sequences

train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
val_dataloader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
# test_dataloader = DataLoader(test_dataset,
#                              batch_size=args.batch_size,
#                              shuffle=False)


# Create model

assert args.model_type in ['convnet', 'resnet', 'transformer', 'phys']
if args.model_type == 'convnet':
    model = NanoporeConvNet(input_size=args.seq_len).to(device)
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
                use_do=False).to(device)
    summary(model, (1, 400))
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
        sample.to(device)
        label.to(device)
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
            sample.to(device)
            label.to(device)
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
            with open(best_model_accuracy_fn, 'w') as accuracy_fn:
                accuracy_fn.write('train_loss\ttrain_acc\tval_loss\tval_acc\n')
                accuracy_fn.write('{train_loss}\t{train_acc}\t{val_loss}\t{val_acc}\n'.format(train_loss = train_loss, train_acc = train_acc, val_loss = val_loss, val_acc = val_acc))
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