#!/usr/bin/env python

import torch
import moviepy
import moviepy.editor
import json
import numpy as np
from cv2 import cv2

import softsplat

# TODO: Implement torch.utils.data.Dataset => InpaintingDataset
# TODO: InpaintingDataset.__getitem__(): get random masked input (image and depth) + get GT (image and depth)
# TODO: InpaintingDataset.__getitem__(): get feature acttivations from VGG-19 relu4_4 for masked input (image and depth), and GT (image and depth)
# TODO: Implement transforms => 

file_image1 = 'images/test/00001-bl-image.png'
file_depth1 = 'images/test/00001-bl-depth.exr'
file_image2 = 'images/test/00001-br-image.png'
file_depth2 = 'images/test/00001-br-depth.exr'

# with open('images/test/00001-meta.json', 'r') as json_file:
#     meta_json = json.loads(json_file.read())

image = cv2.imread(filename=file_image1, flags=cv2.IMREAD_COLOR).transpose(2, 0, 1)
depth = cv2.imread(filename=file_depth1, flags=cv2.IMREAD_ANYDEPTH)[:, :, None].transpose(2, 0, 1)

# max_dim = max(image.shape[1], image.shape[2]) / 2
# fltFov_ = float(meta_json['fltFov']) / 2
# focal = max_dim / np.tan(np.deg2rad(fltFov_))
baseline = 20

tenImage = torch.FloatTensor(np.ascontiguousarray(image[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()
tenDepth = torch.FloatTensor(np.ascontiguousarray(depth[None, :, :, :])).cuda()

### color masking
tenDisp = (image.shape[1] * baseline) / tenDepth
#tenDisp = (focal * baseline) / tenDepth

# -1,0	=> right to left: input order: (bl, br) or (tl, tr)
# 1,0	=> left to right: input order: (br, bl) or (tr, tl)
# 0,1	=> top to bottom: input order: (bl, tl) or (br, tr)
# 0,-1	=> bottom to top: input order: (tl, bl) or (tr, br)
tenFlow = torch.cat([-1.0 * tenDisp, 0.0 * tenDisp], 1)

npyWarped = []
for intTime, fltTime in enumerate(np.linspace(0.0, 1.0, 11).tolist()):
	npyWarped.append((softsplat.FunctionSoftsplat(tenInput=tenImage, tenFlow=tenFlow * fltTime, tenMetric=1.0 + tenDisp, strType='softmax')[0, :, :, :].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8))

moviepy.editor.ImageSequenceClip(sequence=[npyFrame[:, :, ::-1] for npyFrame in npyWarped + list(reversed(npyWarped))], fps=9).write_gif('images/warps/warped.gif')
moviepy.editor.ImageSequenceClip(sequence=[npyFrame[:, :, ::-1] for npyFrame in [npyWarped[-1], cv2.imread(filename=file_image2, flags=-1)]], fps=3).write_gif('images/warps/compare.gif')

image_masked = npyWarped[-1]
cv2.imwrite('images/warps/image_masked.png', image_masked)
image_gt = cv2.imread(filename=file_image2, flags=-1)
cv2.imwrite('images/warps/image_gt.png', image_gt)

### disparity masking
tenDispInput = tenDisp# * (1.0 / 255.0)
depth2 = cv2.imread(filename=file_depth2, flags=cv2.IMREAD_ANYDEPTH)[:, :, None].transpose(2, 0, 1)

tenDepth2 = torch.FloatTensor(np.ascontiguousarray(depth2[None, :, :, :])).cuda()
tenDisp2 = ((image.shape[1] * baseline) / tenDepth2)[0,:,:].cpu().numpy().transpose(1, 2, 0)

npyWarpedDepth = []
for intTime, fltTime in enumerate(np.linspace(0.0, 1.0, 11).tolist()):
	#npyWarpedDepth.append((softsplat.FunctionSoftsplat(tenInput=tenDispInput, tenFlow=tenFlow * fltTime, tenMetric=1.0 + tenDisp, strType='softmax')[0, :, :, :].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8))
	npyWarpedDepth.append((softsplat.FunctionSoftsplat(tenInput=tenDispInput, tenFlow=tenFlow * fltTime, tenMetric=1.0 + tenDisp, strType='softmax')[0, :, :, :].cpu().numpy().transpose(1, 2, 0)))

moviepy.editor.ImageSequenceClip(sequence=[npyFrame[:, :, ::-1] for npyFrame in npyWarpedDepth + list(reversed(npyWarpedDepth))], fps=9).write_gif('images/warps/warped-depth.gif')
moviepy.editor.ImageSequenceClip(sequence=[npyFrame[:, :, ::-1] for npyFrame in [npyWarpedDepth[-1], tenDisp2]], fps=3).write_gif('images/warps/compare-depth.gif')

depth_masked = npyWarpedDepth[-1]
cv2.imwrite('images/warps/depth_masked.png', depth_masked)
depth_gt = tenDisp2
cv2.imwrite('images/warps/depth_gt.png', depth_gt)

# # Extract mask
# mask = (np.any(image_masked != [0, 0, 0], axis=-1).astype(np.uint8) * 255)[:,:,None]
# cv2.imwrite('mask.png', mask)

# # Select values from br image in mask and fill in masked regions in image_masked
# filled_mask = image_gt[mask]
# test = image_gt[:,image_masked]
# filled_mask