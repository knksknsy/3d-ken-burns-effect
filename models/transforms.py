import torch
import torchvision
import cupy
import numpy as np


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
