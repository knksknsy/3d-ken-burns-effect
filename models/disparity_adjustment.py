import torch
import torchvision

moduleMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()

def disparity_adjustment(tenImage, tenDisparity):
	assert(tenImage.shape[0] == 1)
	assert(tenDisparity.shape[0] == 1)

	boolUsed = {}
	tenMasks = []

	objPredictions = moduleMaskrcnn([ tenImage[ 0, [ 2, 0, 1 ], :, : ] ])[0]

	for intMask in range(objPredictions['masks'].shape[0]): # objPredictions['masks'].shape = [4, 1, 768, 1024]
		if intMask in boolUsed:
			continue

		elif objPredictions['scores'][intMask].item() < 0.7:
			continue

		elif objPredictions['labels'][intMask].item() not in [ 1, 3, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 ]: # specific object detected: humans, cars, animals, etc.
			continue

		# end

		boolUsed[intMask] = True # if mask is of specific object => add to adjustment todo list
		tenMask = (objPredictions['masks'][(intMask + 0):(intMask + 1), :, :, :] > 0.5).float() # add threshold to selected mask => 0 or 1 values for mask

		if tenMask.sum().item() < 64: # ignore small masks
			continue
		# end

		for intMerge in range(objPredictions['masks'].shape[0]):
			if intMerge in boolUsed:
				continue

			elif objPredictions['scores'][intMerge].item() < 0.7:
				continue

			elif objPredictions['labels'][intMerge].item() not in [ 2, 4, 27, 28, 31, 32, 33 ]: # specific object detected
				continue

			# end

			tenMerge = (objPredictions['masks'][(intMerge + 0):(intMerge + 1), :, :, :] > 0.5).float() # add threshold to selected mask => 0 or 1 values for mask

			if ((tenMask + tenMerge) > 1.0).sum().item() < 0.03 * tenMerge.sum().item(): # ???
				continue
			# end

			boolUsed[intMerge] = True
			tenMask = (tenMask + tenMerge).clamp(0.0, 1.0) # set values < 0 to 0.0 # set values > 1 to 1.0
		# end

		tenMasks.append(tenMask)
	# end

	tenAdjusted = torch.nn.functional.interpolate(input=tenDisparity, size=(tenImage.shape[2], tenImage.shape[3]), mode='bilinear', align_corners=False) # Resize (upscale) tenDisparity to original tenImage size

	for tenAdjust in tenMasks:
		tenPlane = tenAdjusted * tenAdjust # ???

		tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()
		tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()

		intLeft = (tenPlane.sum(2, True) > 0.0).flatten().nonzero()[0].item() # Get index of first non-zero value of mask (width)
		intTop = (tenPlane.sum(3, True) > 0.0).flatten().nonzero()[0].item() # Get index of first non-zero value of mask (height)
		intRight = (tenPlane.sum(2, True) > 0.0).flatten().nonzero()[-1].item() # Get index of last non-zero value of mask (width)
		intBottom = (tenPlane.sum(3, True) > 0.0).flatten().nonzero()[-1].item() # Get index of last non-zero value of mask (height)

		tenAdjusted = ((1.0 - tenAdjust) * tenAdjusted) + (tenAdjust * tenPlane[:, :, int(round(intTop + (0.97 * (intBottom - intTop)))):, :].max()) # ???
	# end

	return torch.nn.functional.interpolate(input=tenAdjusted, size=(tenDisparity.shape[2], tenDisparity.shape[3]), mode='bilinear', align_corners=False) # Resize (downscale) tenAdjusted to tenDisparity size
# end