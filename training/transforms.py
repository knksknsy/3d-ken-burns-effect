import torch
import torchvision
import numpy as np
from cv2 import cv2

if torch.cuda.is_available():
    import softsplat


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        train_mode, fltFov = sample['train_mode'], sample['fltFov']

        if train_mode == 'inpainting':
            image_from, image_to = sample['image_from'], sample['image_to']
            depth_from, depth_to = sample['depth_from'], sample['depth_to']
            flow = sample['flow']

            image_from = torch.FloatTensor(np.ascontiguousarray(image_from.transpose(2, 0, 1).astype(np.float32)) * (1.0 / 255.0)).to(self.device)
            image_to = torch.FloatTensor(np.ascontiguousarray(image_to.transpose(2, 0, 1).astype(np.float32)) * (1.0 / 255.0)).to(self.device)
            depth_from = torch.FloatTensor(np.ascontiguousarray(depth_from[None, :, :].astype(np.float32))).to(self.device)
            depth_to = torch.FloatTensor(np.ascontiguousarray(depth_to[None, :, :].astype(np.float32))).to(self.device)

            return {'image_from': image_from, 'image_to': image_to, 'depth_from': depth_from, 'depth_to': depth_to, 'flow': flow, 'fltFov': fltFov, 'train_mode': train_mode}

        else:
            image, depth  = sample['image'], sample['depth']
            # swap color axis because
            # numpy image: H x W x C ; torch image: C X H X W
            image = torch.FloatTensor(np.ascontiguousarray(image.transpose(2, 0, 1).astype(np.float32)) * (1.0 / 255.0)).to(self.device)
            depth = torch.FloatTensor(np.ascontiguousarray(depth[None, :, :].astype(np.float32))).to(self.device)

            if train_mode == 'estimation':
                return {'image': image, 'depth': depth, 'fltFov': fltFov}

            elif train_mode == 'refinement':
                depth_adjusted = sample['depth_adjusted']
                depth_adjusted = torch.FloatTensor(np.ascontiguousarray(depth_adjusted[None, :, :].astype(np.float32))).to(self.device)
                return {'image': image, 'depth': depth, 'depth_adjusted': depth_adjusted, 'fltFov': fltFov}


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

class RandomWarp(object):
    """Randomly warp image pairs. Creates masked image and depth for inpaining training."""

    def __call__(self, sample):
        image_from, image_to = sample['image_from'], sample['image_to']
        depth_from, depth_to = sample['depth_from'], sample['depth_to']
        flow, fltFov, train_mode = sample['flow'], sample['fltFov'], sample['train_mode']

        image_from, depth_from = image_from[None,:,:,:], depth_from[None,:,:,:]

        flow = torch.cat([flow[0] * depth_from, flow[1] * depth_from], 1)

        color_warped = []
        for intTime, fltTime in enumerate(np.linspace(0.0, 1.0, 11).tolist()):
            color_warped.append(softsplat.FunctionSoftsplat(tenInput=image_from, tenFlow=flow * fltTime, tenMetric=1.0 + depth_from, strType='softmax'))
        image_masked = color_warped[-1]
        image_gt = image_to

        depth_warped = []
        for intTime, fltTime in enumerate(np.linspace(0.0, 1.0, 11).tolist()):
	        depth_warped.append(softsplat.FunctionSoftsplat(tenInput=depth_from, tenFlow=flow * fltTime, tenMetric=1.0 + depth_from, strType='softmax'))
        depth_masked = depth_warped[-1]
        depth_gt = depth_to

        return {'image_masked': image_masked[0,:,:,:], 'image_gt': image_gt, 'depth_masked': depth_masked[0,:,:,:], 'depth_gt': depth_gt, 'fltFov': fltFov}
