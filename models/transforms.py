import torch
import torchvision
import cupy
import numpy as np
from cv2 import cv2


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        image, depth, fltFov, train_mode = sample['image'], sample['depth'], sample['fltFov'], sample['train_mode']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # normalize image
        image = torch.FloatTensor(np.ascontiguousarray(image.transpose(2, 0, 1).astype(np.float32)) * (1.0 / 255.0)).to(self.device)
        depth = torch.FloatTensor(np.ascontiguousarray(depth[None, :, :].astype(np.float32))).to(self.device)

        if train_mode == 'refinement':
            adjusted = sample['depth_adjusted']
            adjusted = torch.FloatTensor(np.ascontiguousarray(depth[None, :, :].astype(np.float32))).to(self.device)
            return {'image': image, 'depth': depth, 'depth_adjusted': adjusted, 'fltFov': fltFov}

        return {'image': image, 'depth': depth, 'fltFov': fltFov}


class DownscaleDepth(object):
    """Downscale target depth by factor 2"""

    def __call__(self, sample):
        image, depth, fltFov, train_mode = sample['image'], sample['depth'], sample['fltFov'], sample['train_mode']

        intWidth = depth.shape[1]
        intHeight = depth.shape[0]

        fltRatio = float(intWidth) / float(intHeight)

        # Resize dimension to max 256 width or height and keep aspect ratio
        intWidth = min(int(256 * fltRatio), 256)
        intHeight = min(int(256 / fltRatio), 256)

        if train_mode == 'estimation':
            depth = cv2.resize(depth, (intWidth, intHeight), interpolation=cv2.INTER_LINEAR)
            #image = cv2.resize(image, (intWidth, intHeight), interpolation=cv2.INTER_LINEAR)
            return {'image': image, 'depth': depth, 'fltFov': fltFov, 'train_mode': train_mode}

        elif train_mode == 'refinement':
            depth_adjusted = cv2.resize(depth, (intWidth, intHeight), interpolation=cv2.INTER_LINEAR)
            return {'image': image, 'depth': depth, 'depth_adjusted': depth_adjusted, 'fltFov': fltFov, 'train_mode': train_mode}

class RandomRescaleCrop(object):
    """Randomly crop input image to random scales. Crops either top and bottom or right and left with probability of 50 %"""

    batch_process_count = 0
    crop_top_bottom = True
    crop_start = 0
    crop_end = 0

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, sample):
        image, depth, fltFov, train_mode = sample['image'], sample['depth'], sample['fltFov'], sample['train_mode']

        image_height, image_width, _ = image.shape
        depth_height, depth_width = depth.shape
        image_scale = image_height / depth_height

        # apply same cropping to each batch
        if (RandomRescaleCrop.batch_process_count == 0):
            crop_keep = 0.6 # crop only 1 - 0.6 = 0.4 of sample
            RandomRescaleCrop.crop_top_bottom = np.random.choice([True, False], p=[0.5, 0.5])
            RandomRescaleCrop.crop_start = round(np.random.uniform(0.0, crop_keep), 2)
            RandomRescaleCrop.crop_end = round(np.random.uniform(0.0, crop_keep - RandomRescaleCrop.crop_start), 2)

        if RandomRescaleCrop.crop_top_bottom:
            # crop top and bottom
            idx_start_depth = int(RandomRescaleCrop.crop_start * depth_height)
            idx_end_depth = int(RandomRescaleCrop.crop_end * depth_height)
            depth = depth[idx_start_depth:depth_height - idx_end_depth,:]

            idx_start_image = int(idx_start_depth * image_scale)
            idx_end_image = int(idx_end_depth * image_scale)
            image = image[idx_start_image:image_height - idx_end_image,:]
        else:
            # crop right and left
            idx_start_depth = int(RandomRescaleCrop.crop_start * depth_width)
            idx_end_depth = int(RandomRescaleCrop.crop_end * depth_width)
            depth = depth[:,idx_start_depth:depth_width - idx_end_depth]

            idx_start_image = int(idx_start_depth * image_scale)
            idx_end_image = int(idx_end_depth * image_scale)
            image = image[:,idx_start_image:image_width - idx_end_image]

        # reset cropping seed for next batch
        if RandomRescaleCrop.batch_process_count >= self.batch_size - 1:
            RandomRescaleCrop.batch_process_count = 0
        else:
            RandomRescaleCrop.batch_process_count += 1
        
        return {'image': image, 'depth': depth, 'fltFov': fltFov, 'train_mode': train_mode}
