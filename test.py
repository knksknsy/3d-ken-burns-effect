from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from zipfile import ZipFile
from cv2 import cv2


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

def show_images_batch(sample_batched):
    """Show image for a batch of samples."""
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.title('Batch from data_loader')

def main():
    csv_file = 'dataset.csv'
    root_dir = os.path.join('D:', os.path.sep, '3d-ken-burns-dataset')

    image_depth_dataset = ImageDepthDataset(csv_file=csv_file,
                                            root_dir=root_dir,
                                            transform=transforms.Compose([
                                                ToTensor()
                                            ]))

    # No Windows support for multiprocessing on CUDA => num_workers=0
    data_loader = DataLoader(image_depth_dataset, batch_size=4, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, sample_batched['image'].size(), sample_batched['depth'].size(), sample_batched['fltFov'])

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_images_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

if __name__ == "__main__":
    main()
