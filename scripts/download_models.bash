#!/bin/bash

wget --verbose --continue --timestamping --output-document=./../models/disparity_estimation.pytorch http://content.sniklaus.com/kenburns/network-disparity.pytorch
wget --verbose --continue --timestamping --output-document=./../models/disparity_refinement.pytorch http://content.sniklaus.com/kenburns/network-refinement.pytorch
wget --verbose --continue --timestamping --output-document=./../models/pointcloud_inpainting.pytorch http://content.sniklaus.com/kenburns/network-inpainting.pytorch