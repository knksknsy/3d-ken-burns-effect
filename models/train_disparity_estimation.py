from __future__ import print_function, division
from cv2 import cv2
import cupy
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile
import argparse
from disparity_estimation import Disparity

class ImageDepthDataset(Dataset):
    """Image Depth dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory to dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        zip_image_path = os.path.join(
            self.root_dir, self.dataset_frame.iloc[idx, 0])
        zip_depth_path = os.path.join(
            self.root_dir, self.dataset_frame.iloc[idx, 1])
        image_path = self.dataset_frame.iloc[idx, 2]
        depth_path = self.dataset_frame.iloc[idx, 3]

        fltFov = self.dataset_frame.iloc[idx, 4]

        # Read image
        archive_image = ZipFile(zip_image_path, 'r')
        image_bytes = archive_image.read(image_path)
        image = cv2.imdecode(np.frombuffer(
            image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Read depth
        archive_depth = ZipFile(zip_depth_path, 'r')
        depth_bytes = archive_depth.read(depth_path)
        depth = cv2.imdecode(np.frombuffer(
            depth_bytes, np.uint8), cv2.IMREAD_ANYDEPTH)

        sample = {'image': image, 'depth': depth, 'fltFov': fltFov}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, fltFov = sample['image'], sample['depth'], sample['fltFov']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # normalize image
        image = torch.FloatTensor(np.ascontiguousarray(
            image.transpose(2, 0, 1).astype(np.float32)) * (1.0 / 255.0)).cuda()
        depth = torch.FloatTensor(np.ascontiguousarray(
            depth.astype(np.float32))).cuda()

        return {'image': image, 'depth': depth, 'fltFov': fltFov}

def optimizer(parameters, lr, betas):
    return torch.optim.Adam(parameters, lr=lr, betas=betas)

def loss_depth():
    pass

def train(args, model, device, data_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(data_loader):
        print(batch_idx, sample_batched['image'].size(), sample_batched['depth'].size(), sample_batched['fltFov'])

        image, depth = sample_batched['image'], sample_batched['depth']
        optimizer.zero_grad()
        output = model(image)
        #loss = F.nll_loss(output, depth) # TODO: implement and call loss_depth()
        #loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(image)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            if args.dry_run:
                break

def train(model, device, data_loader):
    pass

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch 3d Ken Burns Effect: Disparity Estimation Training')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--b1', type=float, default=0.9, metavar='B1', help='beta 1 for adam optimizer (default: 0.9)')
    parser.add_argument('--b2', type=float, default=0.999, metavar='B2', help='beta 1 for adam optimizer (default: 0.999)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--dataset', action='store', type=str, help='Path to dataset')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    kwargs = {
        'batch_size': args.batch_size
        'num_workers': 0,       # TODO: list CPU cores with $lscpu and use this value for multiprocessing
        'pin_memory': True,     # Speeds-up the transfer of dataset between CPU and GPU
        'shuffle': True
    }

    root_dir = args.dataset
    #root_dir = os.path.join('D:', os.path.sep, '3d-ken-burns-dataset')
    image_depth_dataset = ImageDepthDataset(csv_file='dataset.csv',
                                            root_dir=root_dir,
                                            transform=transforms.Compose([
                                                ToTensor()
                                            ]))
    
    # No Windows support for multiprocessing on CUDA => num_workers=0
    train_loader = DataLoader(image_depth_dataset, **kwargs)
    #test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Disparity().to(device)
    optimizer = optimizer(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "stdl_disparity_estimation.pt")

if __name__ == "__main__":
    main()
