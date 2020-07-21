#!/usr/bin/env python

import torch
import torchvision

from cv2 import cv2
import base64
import cupy
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile
from open3d import *

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

exec(open('common.py', 'r').read())

exec(open('models/disparity_estimation.py', 'r').read())
exec(open('models/disparity_adjustment.py', 'r').read())
exec(open('models/disparity_refinement.py', 'r').read())
exec(open('models/pointcloud_inpainting.py', 'r').read())

##########################################################

arguments_strIn = 'images/doublestrike.jpg'
arguments_strOut = 'images/depth.npy'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

if __name__ == '__main__':
    # Get depth map
    npyImage = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)

    fltFocal = max(npyImage.shape[0], npyImage.shape[1]) / 2.0
    fltBaseline = 40.0

    tenImage = torch.FloatTensor(numpy.ascontiguousarray(npyImage.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()

    tenDisparity = disparity_estimation(tenImage)
    tenDisparity = disparity_refinement(torch.nn.functional.interpolate(input=tenImage, size=(tenDisparity.shape[2] * 4, tenDisparity.shape[3] * 4), mode='bilinear', align_corners=False), tenDisparity)
    tenDisparity = torch.nn.functional.interpolate(input=tenDisparity, size=(tenImage.shape[2], tenImage.shape[3]), mode='bilinear', align_corners=False) * (max(tenImage.shape[2], tenImage.shape[3]) / 256.0)
    tenDepth = (fltFocal * fltBaseline) / (tenDisparity + 0.0000001)

    npyDisparity = tenDisparity[0, 0, :, :].cpu().numpy()
    npyDepth = tenDepth[0, 0, :, :].cpu().numpy()

    cv2.imwrite(filename=arguments_strOut.replace('.npy', '.png'), img=(npyDisparity / fltBaseline * 255.0).clip(0.0, 255.0).astype(numpy.uint8))

    numpy.save(arguments_strOut, npyDepth)

    # Create point cloud from image and depth
    color = open3d.geometry.Image(npyImage)
    depth = open3d.geometry.Image(npyDepth)
    rgbd = create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)

    width, height = int(color.get_max_bound()[0]), int(color.get_max_bound()[1])
    fx, fy = width / 2, height / 2
    cx, cy = fx, fy
    camera = open3d.camera.PinholeCameraIntrinsic(width, height, fx=fx, fy=fy, cx=cx, cy=cy)

    pcd = create_point_cloud_from_rgbd_image(rgbd, camera)

    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    # visualize the point cloud
    draw_geometries([pcd])
# end