import torch

class Basic(torch.nn.Module):
	def __init__(self, strType, intChannels):
		super(Basic, self).__init__()

		if strType == 'relu-conv-relu-conv':
			self.moduleMain = torch.nn.Sequential(
				torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		elif strType == 'conv-relu-conv':
			self.moduleMain = torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		# end

		if intChannels[0] == intChannels[2]:
			self.moduleShortcut = None

		elif intChannels[0] != intChannels[2]:
			self.moduleShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0)

		# end
	# end

	def forward(self, tenInput):
		if self.moduleShortcut is None:
			return self.moduleMain(tenInput) + tenInput

		elif self.moduleShortcut is not None:
			return self.moduleMain(tenInput) + self.moduleShortcut(tenInput)

		# end
	# end
# end

class Downsample(torch.nn.Module):
	def __init__(self, intChannels):
		super(Downsample, self).__init__()

		self.moduleMain = torch.nn.Sequential(
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tenInput):
		return self.moduleMain(tenInput)
	# end
# end

class Upsample(torch.nn.Module):
	def __init__(self, intChannels):
		super(Upsample, self).__init__()

		self.moduleMain = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
			torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
			torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tenInput):
		return self.moduleMain(tenInput)
	# end
# end

class Inpaint(torch.nn.Module):
	def __init__(self):
		super(Inpaint, self).__init__()

		self.moduleContext = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
			torch.nn.PReLU(num_parameters=64, init=0.25),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
			torch.nn.PReLU(num_parameters=64, init=0.25)
		)

		self.moduleInput = Basic('conv-relu-conv', [ 3 + 1 + 64 + 1, 32, 32 ])

		for intRow, intFeatures in [ (0, 32), (1, 64), (2, 128), (3, 256) ]:
			self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
		# end

		for intCol in [ 0, 1 ]:
			self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([ 32, 64, 64 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([ 64, 128, 128 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([ 128, 256, 256 ]))
		# end

		for intCol in [ 2, 3 ]:
			self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([ 256, 128, 128 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([ 128, 64, 64 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([ 64, 32, 32 ]))
		# end

		self.moduleImage = Basic('conv-relu-conv', [ 32, 32, 3 ])
		self.moduleDisparity = Basic('conv-relu-conv', [ 32, 32, 1 ])
	# end

	def forward(self, tenImage, tenDisparity, tenShift):
		tenDepth = (objCommon['fltFocal'] * objCommon['fltBaseline']) / (tenDisparity + 0.0000001)
		tenValid = (spatial_filter(tenDisparity / tenDisparity.max(), 'laplacian').abs() < 0.03).float()
		tenPoints = depth_to_points(tenDepth * tenValid, objCommon['fltFocal'])
		tenPoints = tenPoints.view(1, 3, -1)

		tenMean = [ tenImage.view(tenImage.shape[0], -1).mean(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).mean(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]
		tenStd = [ tenImage.view(tenImage.shape[0], -1).std(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).std(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]

		tenImage = tenImage.clone()
		tenImage -= tenMean[0]
		tenImage /= tenStd[0] + 0.0000001

		tenDisparity = tenDisparity.clone()
		tenDisparity -= tenMean[1]
		tenDisparity /= tenStd[1] + 0.0000001

		tenContext = self.moduleContext(torch.cat([ tenImage, tenDisparity ], 1))

		tenRender, tenExisting = render_pointcloud(tenPoints + tenShift, torch.cat([ tenImage, tenDisparity, tenContext ], 1).view(1, 68, -1), objCommon['intWidth'], objCommon['intHeight'], objCommon['fltFocal'], objCommon['fltBaseline'])

		tenExisting = (tenExisting > 0.0).float()
		tenExisting = tenExisting * spatial_filter(tenExisting, 'median-5')
		tenRender = tenRender * tenExisting.clone().detach()

		# TODO
		# tenContextOut = (tenContext[0, 0:3, :, :].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).astype(numpy.uint8)
		# tenContextOut = cv2.resize(src=tenContextOut, dsize=(objCommon['intWidth'], objCommon['intHeight']), fx=0.0, fy=0.0, interpolation=cv2.INTER_LINEAR)
		# cv2.imwrite(f'./images/autozoom/kbe/no_inpainting/context-{tenShift[0][1].item():.2f}.png', tenContextOut)

		tenColumn = [ None, None, None, None ]

		tenColumn[0] = self.moduleInput(torch.cat([ tenRender, tenExisting ], 1))
		tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
		tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
		tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2])

		intColumn = 1
		for intRow in range(len(tenColumn)):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != 0:
				tenColumn[intRow] += self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow - 1])
			# end
		# end

		intColumn = 2
		for intRow in range(len(tenColumn) -1, -1, -1):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != len(tenColumn) - 1:
				tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

				if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tenColumn[intRow] += tenUp
			# end
		# end

		intColumn = 3
		for intRow in range(len(tenColumn) -1, -1, -1):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != len(tenColumn) - 1:
				tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

				if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tenColumn[intRow] += tenUp
			# end
		# end

		tenImage = self.moduleImage(tenColumn[0])
		tenImage *= tenStd[0] + 0.0000001
		tenImage += tenMean[0]

		tenDisparity = self.moduleDisparity(tenColumn[0])
		tenDisparity *= tenStd[1] + 0.0000001
		tenDisparity += tenMean[1]

		return {
			'tenExisting': tenExisting,
			'tenImage': tenImage.clamp(0.0, 1.0) if self.training == False else tenImage,
			'tenDisparity': torch.nn.functional.threshold(input=tenDisparity, threshold=0.0, value=0.0)
		}
	# end
# end

# load our pretrained model
moduleInpaint = Inpaint().cuda().eval(); moduleInpaint.load_state_dict(torch.load('./models/our_pointcloud_inpainting.pt'))
# load Niklaus et al. pretrained model
# moduleInpaint = Inpaint().cuda().eval(); moduleInpaint.load_state_dict(torch.load('./models/pointcloud_inpainting.pytorch'))

def pointcloud_inpainting(tenImage, tenDisparity, tenShift):
	return moduleInpaint(tenImage, tenDisparity, tenShift)
# end