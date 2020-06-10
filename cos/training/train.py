"""
The main training script for training on synthetic data
"""

import argparse
import multiprocessing
import os

from typing import Dict, List, Tuple, Optional  # pylint: disable=unused-import
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm  # pylint: disable=unused-import

from cos.training.synthetic_dataset import SyntheticDataset
from cos.training.network import CoSNetwork, \
    center_trim, load_pretrain, \
    normalize_input, unnormalize_input


def train_epoch(model: nn.Module, device: torch.device,
                optimizer: optim.Optimizer,
                train_loader: torch.utils.data.dataloader.DataLoader,
                epoch: int, log_interval: int = 20) -> float:
    """
    Train a single epoch.
    """
    # Set the model to training.
    model.train()

    # Training loop
    losses = []
    interval_losses = []

    for batch_idx, (data, label_voice_signals,
                    window_idx) in enumerate(train_loader):
        data = data.to(device)
        label_voice_signals = label_voice_signals.to(device)
        window_idx = window_idx.to(device)

        # Normalize input, each batch item separately
        data, means, stds = normalize_input(data)

        # Reset grad
        optimizer.zero_grad()

        # Run through the model
        valid_length = model.valid_length(data.shape[-1])
        delta = valid_length - data.shape[-1]
        padded = F.pad(data, (delta // 2, delta - delta // 2))

        output_signal = model(padded, window_idx)
        output_signal = center_trim(output_signal, data)

        # Un-normalize
        output_signal = unnormalize_input(output_signal, means, stds)
        output_voices = output_signal[:, 0]

        loss = model.loss(output_voices, label_voice_signals)

        interval_losses.append(loss.item())

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Update the weights
        optimizer.step()

        # Print the loss
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                np.mean(interval_losses)))

            losses.extend(interval_losses)
            interval_losses = []

    return np.mean(losses)


def test_epoch(model: nn.Module, device: torch.device,
               test_loader: torch.utils.data.dataloader.DataLoader,
               log_interval: int = 20) -> float:
    """
    Evaluate the network.
    """
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, label_voice_signals,
                        window_idx) in enumerate(test_loader):
            data = data.to(device)
            label_voice_signals = label_voice_signals.to(device)
            window_idx = window_idx.to(device)

            # Normalize input, each batch item separately
            data, means, stds = normalize_input(data)

            valid_length = model.valid_length(data.shape[-1])
            delta = valid_length - data.shape[-1]
            padded = F.pad(data, (delta // 2, delta - delta // 2))

            # Run through the model
            output_signal = model(padded, window_idx)
            output_signal = center_trim(output_signal, data)

            # Un-normalize
            output_signal = output_signal * stds.unsqueeze(
                3) + means.unsqueeze(3)

            # Un-normalize
            output_signal = unnormalize_input(output_signal, means, stds)
            output_voices = output_signal[:, 0]

            loss = model.loss(output_voices, label_voice_signals)
            test_loss += loss.item()

            if batch_idx % log_interval == 0:
                print("Loss: {}".format(loss))

        test_loss /= len(test_loader)
        print("\nTest set: Average Loss: {:.4f}\n".format(test_loss))

        return test_loss


def train(args: argparse.Namespace):
    """
    Train the network.
    """
    # Load dataset
    data_train = SyntheticDataset(args.train_dir, n_mics=args.n_mics,
                                  sr=args.sr, perturb_prob=1.0,
                                  mic_radius=args.mic_radius)
    data_test = SyntheticDataset(args.test_dir, n_mics=args.n_mics,
                                 sr=args.sr, mic_radius=args.mic_radius)

    # Set up the device and workers.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Using device {}".format('cuda' if use_cuda else 'cpu'))

    # Set multiprocessing params
    num_workers = min(multiprocessing.cpu_count(), args.n_workers)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    } if use_cuda else {}

    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size,
                                              **kwargs)

    # Set up model
    model = CoSNetwork(n_audio_channels=args.n_mics)
    model.to(device)

    # Set up checkpoints
    if not os.path.exists(os.path.join(args.checkpoints_dir, args.name)):
        os.makedirs(os.path.join(args.checkpoints_dir, args.name))

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.decay)

    # Load pretrain
    if args.pretrain_path:
        state_dict = torch.load(args.pretrain_path)
        load_pretrain(model, state_dict)

    # Load the model if `args.start_epoch` is greater than 0. This will load the model from
    # epoch = `args.start_epoch - 1`
    if args.start_epoch is not None:
        assert args.start_epoch > 0, "start_epoch must be greater than 0."
        start_epoch = args.start_epoch
        checkpoint_path = Path(
            args.checkpoints_dir) / "{}.pt".format(start_epoch - 1)
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
    else:
        start_epoch = 0

    # Loss values
    train_losses = []
    test_losses = []

    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_epoch(model, device, optimizer, train_loader,
                                     epoch, args.print_interval)
            torch.save(
                model.state_dict(),
                os.path.join(args.checkpoints_dir, args.name,
                             "{}.pt".format(epoch)))
            print("Done with training, going to testing")
            test_loss = test_epoch(model, device, test_loader,
                                   args.print_interval)
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))

        return train_losses, test_losses

    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:  # pylint: disable=broad-except
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('train_dir', type=str,
                        help="Path to the training dataset")
    parser.add_argument('test_dir', type=str,
                        help="Path to the testing dataset")
    parser.add_argument('--name', type=str, default="multimic_experiment",
                        help="Name of the experiment")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help="Path to the checkpoints")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size")

    # Physical Params
    parser.add_argument('--n_mics', type=int, default=4,
                        help="Number of mics (also number of channels)")
    parser.add_argument('--mic_radius', default=.03231, type=float,
                        help="Radius in meters of the mic array")

    # Training Params
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of epochs")
    parser.add_argument('--lr', type=float, default=3e-4, help="learning rate")
    parser.add_argument('--sr', type=int, default=44100, help="Sampling rate")
    parser.add_argument('--decay', type=float, default=0, help="Weight decay")
    parser.add_argument('--n_workers', type=int, default=16,
                        help="Number of parallel workers")
    parser.add_argument('--print_interval', type=int, default=20,
                        help="Logging interval")

    parser.add_argument('--start_epoch', type=int, default=None,
                        help="Start epoch")
    parser.add_argument('--pretrain_path', type=str,
                        help="Path to pretrained weights")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")

    train(parser.parse_args())
