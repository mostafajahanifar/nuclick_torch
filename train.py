import argparse
import logging
import sys, os
from pathlib import Path
from typing import DefaultDict

import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.dataset_generator import NuclickDataset
from models.loss_functions import get_loss_function
from evaluate import evaluate
from models import UNet, NuClick_NN
from config import DefaultConfig

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create train and validatoin dataset
    if DefaultConfig.dir_val is not None: # create validation set based on dir_val
        train_set = NuclickDataset(DefaultConfig.dir,
                                   phase='train',
                                   scale=DefaultConfig.img_scale,
                                   drop_rate=DefaultConfig.drop_rate,
                                   jitter_range=DefaultConfig.jitter_range,
                                   object_weights=[1, 3],
                                   augment=True)
        n_train = len(train_set)
        val_set = NuclickDataset(DefaultConfig.dir_val,
                                   phase='validation',
                                   scale=DefaultConfig.img_scale,
                                   drop_rate=0,
                                   jitter_range=0,
                                   object_weights=[1, 3],
                                   augment=False)
        n_val = len(val_set)
    else: # create the validation set as a (randomly selected) percentage of training set
        dataset = NuclickDataset(DefaultConfig.dir,
                                 phase='train',
                                 scale=DefaultConfig.img_scale,
                                 drop_rate=DefaultConfig.drop_rate,
                                 jitter_range=DefaultConfig.jitter_range,
                                 object_weights=[1, 3])
        # Split into train / validation partitions
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(DefaultConfig.seed))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='NuClick', resume='allow', anonymous='must')
    experiment.config.update(dict(network=net.net_name, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Network:         {net.net_name}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Loss function:   {DefaultConfig.loss_type}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # Get loss function:
    loss_function = get_loss_function(DefaultConfig.loss_type)
    global_step = 0
    


    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs }', unit='img', ascii=True) as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    masks_pred_ = torch.sigmoid(masks_pred)

                    if loss_function.use_weight() == True:
                        weight = batch['weights'].to(device=device, dtype=torch.float)
                        loss = loss_function.compute_loss(masks_pred_, true_masks, weight)
                    else:
                        loss = loss_function.compute_loss(masks_pred_, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation round
        if n_val > 0:
            histograms = {}
            for tag, value in net.named_parameters():
                tag = tag.replace('/', '.')
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score = evaluate(net, val_loader, device)
            scheduler.step(val_score)

            logging.info('Validation Dice score: {}'.format(val_score))
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Dice': val_score,
                'images': wandb.Image(images[0, :3, :, :].cpu()),
                'masks': {
                    'true': wandb.Image(true_masks[0].float().cpu()),
                    'pred': wandb.Image(torch.sigmoid(masks_pred)[0, 0, :, :].float().cpu()),
                },
                'step': global_step,
                'epoch': epoch,
                **histograms
            })

        if save_checkpoint:
            Path(DefaultConfig.dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(DefaultConfig.dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=DefaultConfig.epochs, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=DefaultConfig.batch_size, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=DefaultConfig.lr,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=DefaultConfig.model_path, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=DefaultConfig.img_scale, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=DefaultConfig.val_percent,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=DefaultConfig.use_amp, help='Use mixed precision')
    parser.add_argument('--gpu', '-g', metavar='GPU', default=DefaultConfig.gpu, help='ID of GPUs to use (based on `nvidia-smi`)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # setting gpus
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if DefaultConfig.network.lower() == 'unet':
        net = UNet(n_channels=5, n_classes=1, bilinear=True)
    elif DefaultConfig.network.lower() == 'nuclick':
        net = NuClick_NN(n_channels=5, n_classes=1)
    else:
        raise ValueError(f'Unknown network architecture: {DefaultConfig.network}')

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 )

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)