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
import json
import open3d

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

arguments_image = ""
arguments_depth = ""
arguments_meta = ""

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--image' and strArgument != '': arguments_image = strArgument # path to the input image
    if strOption == '--depth' and strArgument != '': arguments_depth = strArgument # path to where the output should be stored
    if strOption == '--meta' and strArgument != '': arguments_meta = strArgument
# end

##########################################################

def get_flattened_pcds2(source, A, B, C, D, x0, y0, z0):
    x1 = numpy.asarray(source.points)[:,0]
    y1 = numpy.asarray(source.points)[:,1]
    z1 = numpy.asarray(source.points)[:,2]
    x0 = x0 * numpy.ones(x1.size)
    y0 = y0 * numpy.ones(y1.size)
    z0 = z0 * numpy.ones(z1.size)
    r = numpy.power(numpy.square(x1-x0) + numpy.square(y1-y0) + numpy.square(z1-z0), 0.5)
    a = (x1 - x0) / r
    b = (y1 - y0)  /r
    c = (z1 - z0) / r
    t = -1 * (A * numpy.asarray(source.points)[:,0] + B * numpy.asarray(source.points)[:,1] + C * numpy.asarray(source.points)[:,2] + D)
    t = t / (a * A + b * B + c * C)
    numpy.asarray(source.points)[:,0] = x1 + a * t
    numpy.asarray(source.points)[:,1] = y1 + b * t
    numpy.asarray(source.points)[:,2] = z1 + c * t
    return source

if __name__ == '__main__':
    # Get depth map
    npyImage = cv2.imread(filename=arguments_image, flags=cv2.IMREAD_COLOR)

    if len(arguments_meta) > 0:
        with open(arguments_meta, 'r') as json_file:
            meta_json = json.loads(json_file.read())
        max_dim = max(npyImage.shape[0], npyImage.shape[1]) / 2
        fltFov_ = float(meta_json['fltFov']) / 2
        fltFocal = max_dim / numpy.tan(numpy.deg2rad(fltFov_))
    else:
        fltFocal = max(npyImage.shape[0], npyImage.shape[1]) / 2

    fltBaseline = 40.0

    tenImage = torch.FloatTensor(numpy.ascontiguousarray(npyImage.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()

    if len(arguments_depth) == 0:
        tenDisparity = disparity_estimation(tenImage)
        tenDisparity = disparity_refinement(torch.nn.functional.interpolate(input=tenImage, size=(tenDisparity.shape[2] * 4, tenDisparity.shape[3] * 4), mode='bilinear', align_corners=False), tenDisparity)
        tenDisparity = torch.nn.functional.interpolate(input=tenDisparity, size=(tenImage.shape[2], tenImage.shape[3]), mode='bilinear', align_corners=False) * (max(tenImage.shape[2], tenImage.shape[3]) / 256.0)
        tenDepth = (fltFocal * fltBaseline) / (tenDisparity + 0.0000001)
        npyDepth = tenDepth[0, 0, :, :].cpu().numpy()
    else: # load depth from file
        npyDepth = cv2.imread(filename=arguments_depth, flags=cv2.IMREAD_ANYDEPTH)
        npyDepth = (fltFocal * fltBaseline) / (npyDepth + 0.0000001)

    # Create point cloud from image and depth
    color = open3d.geometry.Image(npyImage)
    depth = open3d.geometry.Image(npyDepth)
    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1000.0, convert_rgb_to_intensity=False)

    width, height = int(color.get_max_bound()[0]), int(color.get_max_bound()[1])
    fx, fy = width / 2, height / 2
    cx, cy = fx, fy
    camera = open3d.camera.PinholeCameraIntrinsic(width, height, fx=fx, fy=fy, cx=cx, cy=cy)

    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)

    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])

    ## Project to plane
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # pcd = get_flattened_pcds2(pcd, a, b, c, d, 0, 0, 0)

    # visualize the point cloud
    open3d.visualization.draw_geometries([pcd])
# end