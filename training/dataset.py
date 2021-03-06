import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision
import numpy as np
import pandas as pd
import os
import copy
from zipfile import ZipFile
from cv2 import cv2


class ImageDepthDataset(Dataset):
    """Image Depth dataset."""

    # counter assures that exactly same transformations will be used for each batch
    batch_process_count = 0

    def __init__(self, csv_file, dataset_path, train_mode='estimation', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            dataset_path (string): Directory to dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset_frame = pd.read_csv(csv_file)
        self.dataset_path = dataset_path
        self.transform = transform
        self.train_mode = train_mode

    def get_train_valid_loader(self, batch_size, valid_batch_size, valid_size, seed, num_workers, pin_memory):
        """
        Args:
            batch_size (int): Batch size for training set
            valid_batch_size (int): Batch size for validation set
            valid_size (float): Size of validation set
            seed (int): Numpy seed for reproducability
            num_workers (int): Number of workers for multiprocessing. Disabled on Windows => num_workers=0
            pin_memory (bool): Speeds-up the transfer of dataset between CPU and GPU
        """
        self.batch_size = batch_size
        train_dataset = self
        valid_dataset = copy.deepcopy(train_dataset)
        num_train = len(train_dataset)
        indices = list(range(num_train))
        assert ((valid_size >= 0) and (valid_size <= 1)), 'valid-size should be in the range [0, 1].'
        split = int(np.floor(valid_size * num_train))
        # randomly shuffle the dataset
        np.random.seed(seed)
        np.random.shuffle(indices)

        # split the dataset into training and validation sets
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # initialize training and validation DataLoaders used in the training loop to read batches of samples
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, valid_loader

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.dataset_frame)

    def __getitem__(self, idx):
        """
        This method will be called during the training loop.
        E.g.: `for batch_idx, sample_batched in enumerate(data_loader)`.

        `data_loader` can be initialized by calling the `get_train_valid_loader()` method like:

        `transform = transforms.Compose([DownscaleDepth(), RandomRescaleCrop(args.batch_size), ToTensor(device)])`

        `dataset = ImageDepthDataset(csv_file='dataset.csv', dataset_path=args.dataset_path, train_mode='estimation', transform=transform)`

        `train_loader, valid_loader = dataset.get_train_valid_loader(args.batch_size, args.valid_batch_size, args.valid_size, args.seed, args.num_workers, args.pin_memory)`
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get paths to virtual environments
        zip_image_path = os.path.join(self.dataset_path, self.dataset_frame.iloc[idx, 0])
        zip_depth_path = os.path.join(self.dataset_path, self.dataset_frame.iloc[idx, 1])
        baseline = 20

        if self.train_mode == 'estimation' or self.train_mode == 'refinement':
            # get paths for image and its depth map
            image_path = self.dataset_frame.iloc[idx, 2]
            depth_path = self.dataset_frame.iloc[idx, 3]

            fltFov = self.dataset_frame.iloc[idx, 4]

            # Read image
            archive_image = ZipFile(zip_image_path, 'r')
            image_bytes = archive_image.read(image_path)
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Read depth
            archive_depth = ZipFile(zip_depth_path, 'r')
            depth_bytes = archive_depth.read(depth_path)
            depth = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_ANYDEPTH)

            # Convert 32bit depth values to float values
            max_dim = max(image.shape[0], image.shape[1]) / 2
            fltFov_ = fltFov / 2
            focal = max_dim / np.tan(np.deg2rad(fltFov_))
            depth = (focal * baseline) / (depth + 0.0000001)

            sample = {'image': image, 'depth': depth, 'fltFov': fltFov, 'train_mode': self.train_mode}

        elif self.train_mode == 'inpainting':
            # get column ids for images and its depth maps (4 views per image and depth map: TL, TR, BL, BR)
            tl_idx, tr_idx, bl_idx, br_idx = [2,6], [3,7], [4,8], [5,9]
            fltFov = self.dataset_frame.iloc[idx, 10]

            # all possible warping directions
            warpings = [
                {'from': tl_idx, 'to': tr_idx, 'flow': [-1.0, 0.0]},
                {'from': bl_idx, 'to': br_idx, 'flow': [-1.0, 0.0]},
                {'from': tr_idx, 'to': tl_idx, 'flow': [1.0, 0.0]},
                {'from': br_idx, 'to': bl_idx, 'flow': [1.0, 0.0]},
                {'from': bl_idx, 'to': tl_idx, 'flow': [0.0, 1.0]},
                {'from': br_idx, 'to': tr_idx, 'flow': [0.0, 1.0]},
                {'from': tl_idx, 'to': bl_idx, 'flow': [0.0, -1.0]},
                {'from': tr_idx, 'to': br_idx, 'flow': [0.0, -1.0]}
            ]

            # randomly choose one warping direction
            # use the same warping direction for each sample in a batch 
            if (self.batch_process_count == 0):
                self.warping_direction_idx = np.random.randint(0, len(warpings))
            warping_direction = warpings[self.warping_direction_idx]

            # read imageFrom and imageTo
            archive_image = ZipFile(zip_image_path, 'r')
            image_from_path = self.dataset_frame.iloc[idx, warping_direction['from'][0]]
            image_from_bytes = archive_image.read(image_from_path)
            image_from = cv2.imdecode(np.frombuffer(image_from_bytes, np.uint8), cv2.IMREAD_COLOR)
            image_to_path = self.dataset_frame.iloc[idx, warping_direction['to'][0]]
            image_to_bytes = archive_image.read(image_to_path)
            image_to = cv2.imdecode(np.frombuffer(image_to_bytes, np.uint8), cv2.IMREAD_COLOR)

            # read depthFrom and depthTo
            archive_depth = ZipFile(zip_depth_path, 'r')
            depth_from_path = self.dataset_frame.iloc[idx, warping_direction['from'][1]]
            depth_from_bytes = archive_depth.read(depth_from_path)
            depth_from = cv2.imdecode(np.frombuffer(depth_from_bytes, np.uint8), cv2.IMREAD_ANYDEPTH)
            depth_to_path = self.dataset_frame.iloc[idx, warping_direction['to'][1]]
            depth_to_bytes = archive_depth.read(depth_to_path)
            depth_to = cv2.imdecode(np.frombuffer(depth_to_bytes, np.uint8), cv2.IMREAD_ANYDEPTH)

            # Convert 32bit depth values to float values
            depth_from = (depth_from.shape[1] * baseline) / (depth_from)
            depth_to = (depth_to.shape[1] * baseline) / (depth_to)

            if self.batch_process_count >= self.batch_size -1:
                self.batch_process_count = 0
            else:
                self.batch_process_count += 1

            sample = {'image_from': image_from, 'image_to': image_to, 'depth_from': depth_from, 'depth_to': depth_to, 'flow': warping_direction['flow'], 'fltFov': fltFov, 'train_mode': self.train_mode}

        # apply transformations (see training/transforms.py)
        if self.transform:
            sample = self.transform(sample)

        return sample
