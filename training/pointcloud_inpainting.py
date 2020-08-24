import torch

device = 'cpu'
if torch.cuda.is_available():
    import cupy
    device = torch.device('cuda')

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

	def forward(self, tenImageMasked, tenImage, tenDisparityMasked, tenDisparity):
		tenMean = [ tenImage.view(tenImage.shape[0], -1).mean(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).mean(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]
		tenStd = [ tenImage.view(tenImage.shape[0], -1).std(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).std(1, True).view(tenDisparity.shape[0], 1, 1, 1) ]

		tenImage = tenImage.clone()
		tenImage -= tenMean[0]
		tenImage /= tenStd[0] + 0.0000001

		tenDisparity = tenDisparity.clone()
		tenDisparity -= tenMean[1]
		tenDisparity /= tenStd[1] + 0.0000001

		tenContext = self.moduleContext(torch.cat([ tenImage, tenDisparity ], 1))
		tenRender = torch.cat([tenImageMasked, tenDisparityMasked, tenContext], 1)

		tenMask = (tenDisparityMasked > 0.0).float()
		tenMask = tenMask * spatial_filter(tenMask, 'median-5')
		tenRender = tenRender * tenMask.detach().clone()

		tenColumn = [ None, None, None, None ]

		tenColumn[0] = self.moduleInput(torch.cat([tenRender, tenMask], 1))
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

		return tenImage, torch.nn.functional.threshold(input=tenDisparity, threshold=0.0, value=0.0)
	# end
# end

def init_weights(m):
	if type(m) == torch.nn.Conv2d:
		torch.nn.init.xavier_uniform_(m.weight)
		torch.nn.init.constant_(m.bias, 0.0)

def spatial_filter(tenInput, strType):
	tenOutput = None

	if strType == 'laplacian':
		tenLaplacian = tenInput.new_zeros(tenInput.shape[1], tenInput.shape[1], 3, 3)

		for intKernel in range(tenInput.shape[1]):
			tenLaplacian[intKernel, intKernel, 0, 1] = -1.0
			tenLaplacian[intKernel, intKernel, 0, 2] = -1.0
			tenLaplacian[intKernel, intKernel, 1, 1] = 4.0
			tenLaplacian[intKernel, intKernel, 1, 0] = -1.0
			tenLaplacian[intKernel, intKernel, 2, 0] = -1.0
		# end

		tenOutput = torch.nn.functional.pad(input=tenInput, pad=[ 1, 1, 1, 1 ], mode='replicate')
		tenOutput = torch.nn.functional.conv2d(input=tenOutput, weight=tenLaplacian)

	elif strType == 'median-3':
		tenOutput = torch.nn.functional.pad(input=tenInput, pad=[ 1, 1, 1, 1 ], mode='reflect')
		tenOutput = tenOutput.unfold(2, 3, 1).unfold(3, 3, 1)
		tenOutput = tenOutput.contiguous().view(tenOutput.shape[0], tenOutput.shape[1], tenOutput.shape[2], tenOutput.shape[3], 3 * 3)
		tenOutput = tenOutput.median(-1, False)[0]

	elif strType == 'median-5':
		tenOutput = torch.nn.functional.pad(input=tenInput, pad=[ 2, 2, 2, 2 ], mode='reflect')
		tenOutput = tenOutput.unfold(2, 5, 1).unfold(3, 5, 1)
		tenOutput = tenOutput.contiguous().view(tenOutput.shape[0], tenOutput.shape[1], tenOutput.shape[2], tenOutput.shape[3], 5 * 5)
		tenOutput = tenOutput.median(-1, False)[0]

	# end

	return tenOutput
# end

# if torch.cuda.is_available():
# 	moduleInpaint = Inpaint().to(device).eval(); moduleInpaint.load_state_dict(torch.load('./models/pointcloud_inpainting.pytorch'))
# else:
# 	moduleInpaint = Inpaint().to(device).eval(); moduleInpaint.load_state_dict(torch.load('./models/pointcloud_inpainting.pytorch', map_location=torch.device('cpu')))

def pointcloud_inpainting(tenImage, tenDisparity, tenShift):
	return moduleInpaint(tenImage, tenDisparity, tenShift)
# end