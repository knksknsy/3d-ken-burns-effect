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

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

exec(open('./common.py', 'r').read())

exec(open('./models/disparity_estimation.py', 'r').read())
exec(open('./models/disparity_adjustment.py', 'r').read())
exec(open('./models/disparity_refinement.py', 'r').read())
exec(open('./models/pointcloud_inpainting.py', 'r').read())

##########################################################

arguments_strIn = './images/doublestrike.jpg'
arguments_strOut = './depthestim.npy'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

if __name__ == '__main__':
	npyImage = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)

	fltFocal = max(npyImage.shape[0], npyImage.shape[1]) / 2.0
	fltBaseline = 40.0

	tenImage = torch.FloatTensor(numpy.ascontiguousarray(npyImage.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
	
	tenDisparity = disparity_estimation(tenImage)
	print(f'estimation.shape:\n{tenDisparity.shape}')
	print(f'estimation:\n{tenDisparity}')
	#cv2.imshow('Estimation', tenDisparity[0,0,:,:].cpu().numpy())
	#cv2.waitKey()
	
	tenDisparity = disparity_refinement(torch.nn.functional.interpolate(input=tenImage, size=(tenDisparity.shape[2] * 4, tenDisparity.shape[3] * 4), mode='bilinear', align_corners=False), tenDisparity)
	print(f'refinement.shape:\n{tenDisparity.shape}')
	print(f'refinement:\n{tenDisparity}')
	#cv2.imshow('Refinement', tenDisparity[0,0,:,:].cpu().numpy())
	#cv2.waitKey()
	
	tenDisparity = torch.nn.functional.interpolate(input=tenDisparity, size=(tenImage.shape[2], tenImage.shape[3]), mode='bilinear', align_corners=False) * (max(tenImage.shape[2], tenImage.shape[3]) / 256.0)
	tenDepth = (fltFocal * fltBaseline) / (tenDisparity + 0.0000001)

	npyDisparity = tenDisparity[0, 0, :, :].cpu().numpy()
	print(f'npyDisparity.shape:\n{npyDisparity.shape}')
	print(f'npyDisparity:\n{npyDisparity}')
	#cv2.imshow('npyDisparity', npyDisparity)
	#cv2.waitKey()
	
	npyDepth = tenDepth[0, 0, :, :].cpu().numpy()

	print(f'npyDepth:\n{npyDepth}')

	cv2.imwrite(filename=arguments_strOut.replace('.npy', '.png'), img=(npyDisparity / fltBaseline * 255.0).clip(0.0, 255.0).astype(numpy.uint8))

	numpy.save(arguments_strOut, npyDepth)
# end