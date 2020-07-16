import torch
import torchvision
import cupy
import numpy as np
from cv2 import cv2


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, fltFov = sample['image'], sample['depth'], sample['fltFov']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # normalize image
        image = torch.FloatTensor(np.ascontiguousarray(image.transpose(2, 0, 1).astype(np.float32)) * (1.0 / 255.0)).cuda()
        depth = torch.FloatTensor(np.ascontiguousarray(depth[None, :, :].astype(np.float32))).cuda()

        return {'image': image, 'depth': depth, 'fltFov': fltFov}


class DownscaleDepth(object):
    """Downscale target depth by factor 2"""

    def __call__(self, sample):
        image, depth, fltFov = sample['image'], sample['depth'], sample['fltFov']

        intWidth = depth.shape[1]
        intHeight = depth.shape[0]

        fltRatio = float(intWidth) / float(intHeight)

        # Resize dimension to max 256 width or height and keep aspect ratio
        intWidth = min(int(256 * fltRatio), 256)
        intHeight = min(int(256 / fltRatio), 256)

        depth = cv2.resize(depth, (intWidth, intHeight),
                           interpolation=cv2.INTER_LINEAR)

        return {'image': image, 'depth': depth, 'fltFov': fltFov}
