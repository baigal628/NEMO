import argparse
import pandas as pd
from tqdm import tqdm
import wandb

# pytorch
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

# data set formatting
from nanopore_dataset import create_sample_map
from nanopore_dataset import create_splits
from nanopore_dataset import load_sigalign
from nanopore_dataset import load_parquet
from nanopore_dataset import NanoporeDataset

# models
from resnet1d import ResNet1D
from nanopore_convnet import NanoporeConvNet
from nanopore_transformer import NanoporeTransformer




#################################
# train, validation, test split #
#################################


def train_test_split(pos_data, neg_data, input_dtype, seq_len, min_val, max_val, max_seqs, train_split, val_split, test_split, device, exp_id, model_type):
    if input_dtype == 'sigalign':
        load_data = load_sigalign
    elif input_dtype == 'parquet':
        load_data = load_parquet

    print("Preparing unmodified...")
    print("Loading csv...")
    unmodified_sequences = load_data(neg_data,
                                    min_val=min_val,
                                    max_val=max_val,
                                    max_sequences=max_seqs)
    print("Creating sample map...")
    unmodified_sample_map = create_sample_map(unmodified_sequences,
                                            seq_len=seq_len)

    print("Creating splits...")
    unmodified_train, unmodified_val, unmodified_test = create_splits(unmodified_sequences,
                                                                    unmodified_sample_map,
                                                                    train_split=train_split,
                                                                    val_split=val_split,
                                                                    test_split=test_split,
                                                                    shuffle=True,
                                                                    seq_len=seq_len)
    print("Prepared.")

    print("Preparing modified...")
    print("Loading csv...")
    modified_sequences = load_data(pos_data,
                                    min_val=min_val,
                                    max_val=max_val,
                                    max_sequences=max_seqs)
    print("Creating sample map...")
    modified_sample_map = create_sample_map(modified_sequences,
                                            seq_len=seq_len)
    print("Creating splits...")
    modified_train, modified_val, modified_test = create_splits(modified_sequences,
                                                                modified_sample_map,
                                                                train_split=train_split,
                                                                val_split=val_split,
                                                                test_split=test_split,
                                                                shuffle=True,
                                                                seq_len=seq_len)
    print("Prepared.")

    ###############################
    # create torch data set class #
    ###############################

    print('Creating torch dataset class...')
    train_dataset = NanoporeDataset(unmodified_sequences,
                                    unmodified_train,
                                    modified_sequences,
                                    modified_train,
                                    device=device,
                                    synthetic=False,
                                    seq_len=seq_len)

    val_dataset = NanoporeDataset(unmodified_sequences,
                                unmodified_val,
                                modified_sequences,
                                modified_val,
                                device=device,
                                synthetic=False,
                                seq_len=seq_len)

    test_dataset = NanoporeDataset(unmodified_sequences,
                                unmodified_test,
                                modified_sequences,
                                modified_test,
                                device=device,
                                synthetic=False,
                                seq_len=seq_len)
    # test_dataloader = DataLoader(test_dataset,
    #                              batch_size=args.batch_size,
    #                              shuffle=True)

    if args.save_test:
        torch.save(val_dataset, f'{args.outpath}/val_dataset_{exp_id}_{model_type}.pt')
        torch.save(train_dataset, f'{args.outpath}/train_dataset_{exp_id}_{model_type}.pt')
        torch.save(test_dataset, f'{args.outpath}/test_dataset_{exp_id}_{model_type}.pt')
    
    return train_dataset, val_dataset

################
# set up model #
################

def train(train_dataloader, val_dataloader, exp_id, model_type, opt, seq_len, device, epochs, steps_per_epoch, val_steps_per_epoch, batch_size, 
          lr, decay, momentum, patience, wandb_id, best_model_fn, best_model_accuracy_fn, metrics_fn):

    if wandb_id:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_id,

            # track hyperparameters and run metadata
            config={
            "architecture": model_type,
            "learning_rate": lr,
            "dataset": exp_id,
            "epochs": epochs,
            "batch_size:": batch_size,
            "opti:": opt,
            "momentum:": momentum,
            "decay:": decay

            }
        )
        
    assert model_type in ['convnet', 'resnet', 'transformer', 'phys']

    if model_type == 'convnet':
        model = NanoporeConvNet(input_size=seq_len).to(device)
    elif model_type == 'resnet':
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
    elif model_type == 'transformer':
        model = NanoporeTransformer(d_model=128,
                                    dim_feedforward=256,
                                    n_layers=6,
                                    n_head=8).to(device)
    elif model_type == 'phys':
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
    
    summary(model, (1, seq_len))
    print("Created model and moved to the device.")

    # Create loss function and optimizer
    loss_function = nn.BCEWithLogitsLoss()

    assert opt in ['adam', 'sgd']
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                            lr=lr,
                            weight_decay=decay,
                            momentum=momentum,
                            nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode='max',
                                                    factor=0.1,
                                                    patience=patience)


    ##################
    # training model #
    ##################

    print('Start training...')

    best_val_acc = -1
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):

        model.train()

        # Set up progress bar
        print()
        progress = tqdm(range(steps_per_epoch),
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

            if step == steps_per_epoch:
                break


        # Evaluate on validation set

        model.eval()
        with torch.no_grad():

            # Set up progress bar
            print()
            progress = tqdm(range(val_steps_per_epoch),
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
                if step == val_steps_per_epoch:
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
            if wandb_id:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc})
            
    metrics_df = pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs})
    metrics_df.to_csv(metrics_fn)

def add_parser(parser):
    parser.add_argument('--exp_id', default='test')
    parser.add_argument('--wandb_id', default='')
    parser.add_argument('--device', default='auto')

    # if you do not have a train validation split, specify the below
    parser.add_argument('--neg_data', type= str, default='', help='unmodified dechromatanized dna sequences, either a siglaign file or a parquet file format.')
    parser.add_argument('--pos_data', type= str, default='', help='fully modified dechromatinized dna sequences, either a siglaign file or a parquet file format.')
    parser.add_argument('--input_dtype', type= str, default='parquet', help='choose between sigalign or parquet. DEFAULT: parquet.')
    parser.add_argument('--train_split', type=float, default=0.6, help='fraction of data used for training model. DEFAULT: 0.6.')
    parser.add_argument('--val_split', type=float, default=0.2, help='fraction of data used for model validation. DEFAULT: 0.2.')
    parser.add_argument('--test_split', type=float, default=0.2, help='fraction of data used for testing model. DEFAULT: 0.2.')
    parser.add_argument('--save_test', action='store_true', help='save the test dataset for further validation.')

    # if you have a train validation split, specify the below
    parser.add_argument('--train_dataset', type= str, default='', help='custom pytorch dataset for training.')
    parser.add_argument('--val_dataset', type= str, default='', help='custom pytorch dataset for validation.')

    #input and output data preprocessing parameters
    parser.add_argument('--min_val', type=float, default=50) # Used to clip outliers
    parser.add_argument('--max_val', type=float, default=130) # Used to clip outliers
    parser.add_argument('--seq_len', type=int, default=400)
    parser.add_argument('--max_seqs', type=int, default=None)
    parser.add_argument('--outpath', type=str, default='./')

    # model paramters
    parser.add_argument('--model_type', default='resnet')

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--val_steps_per_epoch', type=int, default=500)
    parser.add_argument('--opt', default='adam', help='optimizer to use. choose between [sgd, adam]')
    parser.add_argument('--decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train neural network model on nanopore signal data.')
    add_parser(parser)
    args = parser.parse_args()

    ##############
    # set device #
    ##############
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print('Device type:', device)

    if not args.train_dataset:
        print('Splitting data into train, test, and validation dataset.')
        tstart = time.time()
        train_dataset, val_dataset = train_test_split(pos_data=args.pos_data, 
                                                      neg_data=args.neg_data, 
                                                      input_dtype=args.input_dtype,
                                                      seq_len=args.seq_len,
                                                      min_val=args.min_val,
                                                      max_val= args.max_val,
                                                      max_seqs=args.max_seqs,
                                                      train_split=args.train_split,
                                                      val_split=args.val_split,
                                                      test_split=args.test_split,
                                                      device=device,
                                                      exp_id=args.exp_id,
                                                      model_type=args.model_type)
        print(f'Finished splitting data in {round(time.time()-tstart, 3)} sec.')
    else:
        print("Loading user defined data...")
        tstart = time.time()
        train_dataset = torch.load(args.train_dataset)
        val_dataset = torch.load(args.val_dataset)
        print(f'Finished loading user defined data in {round(time.time()-tstart, 3)} sec.')
    
    train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)
    
    ###############
    # set outfile #
    ###############
    # best model based on validation accuracy
    best_model_fn = f'{args.outpath}/{args.exp_id}_{args.model_type}_best_model.pt'
    # stats associated with best model
    best_model_accuracy_fn = f'{args.outpath}/{args.exp_id}_{args.model_type}_best_model.csv'
    # stats associated with best model
    metrics_fn = f'{args.outpath}/{args.exp_id}_{args.model_type}.csv'

    print('Start training...')
    tstart = time.time()
    train(train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          exp_id=args.exp_id,
          model_type=args.model_type, 
          opt=args.opt,
          seq_len=args.seq_len, 
          device=device,
          epochs=args.epochs,
          steps_per_epoch=args.steps_per_epoch,
          val_steps_per_epoch=args.val_steps_per_epoch,
          batch_size=args.batch_size, 
          lr=args.lr,
          decay=args.decay,
          momentum=args.momentum,
          patience=args.patience,
          wandb_id=args.wandb_id,
          best_model_fn=best_model_fn,
          best_model_accuracy_fn=best_model_accuracy_fn,
          metrics_fn=metrics_fn)
    
    print(f'Training completed in {round(time.time()-tstart, 3)} sec.')
    