import torch
import torchvision
import numpy as np
from cv2 import cv2
import json

# Create depth map
with open('test/00001-meta.json', 'r') as json_file:
    meta_json = json.loads(json_file.read())

depth1 = cv2.imread('test/00001-bl-depth.exr', cv2.IMREAD_ANYDEPTH)
depth2 = cv2.imread('test/00001-br-depth.exr', cv2.IMREAD_ANYDEPTH)
depth3 = cv2.imread('test/00001-tl-depth.exr', cv2.IMREAD_ANYDEPTH)
depth4 = cv2.imread('test/00001-tr-depth.exr', cv2.IMREAD_ANYDEPTH)

image_shape = depth1.shape

baseline = 20
max_dim = max(image_shape[0], image_shape[1]) / 2
fltFov_ = float(meta_json['fltFov']) / 2
focal = max_dim / np.tan(np.deg2rad(fltFov_))
depth1 = (focal * baseline) / (depth1 + 0.0000001)
depth2 = (focal * baseline) / (depth2 + 0.0000001)
depth3 = (focal * baseline) / (depth3 + 0.0000001)
depth4 = (focal * baseline) / (depth4 + 0.0000001)

cv2.imwrite('test/output/00001-bl-depth.png', depth1.clip(0.0, 255.0).astype(np.uint8))
cv2.imwrite('test/output/00001-br-depth.png', depth2.clip(0.0, 255.0).astype(np.uint8))
cv2.imwrite('test/output/00001-tl-depth.png', depth3.clip(0.0, 255.0).astype(np.uint8))
cv2.imwrite('test/output/00001-tr-depth.png', depth4.clip(0.0, 255.0).astype(np.uint8))

# # Render point cloud from image and depth
# from open3d import *

# color = cv2.imread('test-image.png', cv2.IMREAD_COLOR)
# color = open3d.geometry.Image(color)
# #depth = cv2.imread('test-depth.png', cv2.IMREAD_ANYDEPTH)
# depth = np.load('test-depth.npy')
# depth = open3d.geometry.Image(depth)

# rgbd = create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)

# width, height = int(color.get_max_bound()[0]), int(color.get_max_bound()[1])
# fx, fy = width / 2, height / 2
# cx, cy = fx, fy
# camera = open3d.camera.PinholeCameraIntrinsic(width, height, fx=fx, fy=fy, cx=cx, cy=cy)

# pcd = create_point_cloud_from_rgbd_image(rgbd, camera)

# # flip the orientation, so it looks upright, not upside-down
# pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

# draw_geometries([pcd])    # visualize the point cloud