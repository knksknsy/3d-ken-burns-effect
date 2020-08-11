import torch
import torchvision

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
				torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1), # in_channels=32, out_channels=32
				torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), # num_parameters=32
				torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1) # in_channels=32, out_channels=1
			)

		# end

		if intChannels[0] == intChannels[2]:
			self.moduleShortcut = None

		elif intChannels[0] != intChannels[2]:
			self.moduleShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0) # in_channels=32, out_channels=1

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

class Semantics(torch.nn.Module):
	def __init__(self):
		super(Semantics, self).__init__()

		moduleVgg = torchvision.models.vgg19_bn(pretrained=True).features.eval()

		# adjust vgg19 architecture: get first layers 0 - 40 (original #layers 53) # replace MaxPool2d(..., ceil_mode=True) with MaxPool2d(..., ceil_mode=False) => aufrunden statt abrunden
		self.moduleVgg = torch.nn.Sequential(
			moduleVgg[0:3],
			moduleVgg[3:6],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			moduleVgg[7:10],
			moduleVgg[10:13],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			moduleVgg[14:17],
			moduleVgg[17:20],
			moduleVgg[20:23],
			moduleVgg[23:26],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			moduleVgg[27:30],
			moduleVgg[30:33],
			moduleVgg[33:36],
			moduleVgg[36:39],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
		)
	# end

	def forward(self, tenInput):
		# Reverse order of channel
		tenPreprocessed = tenInput[:, [ 2, 1, 0 ], :, :]

		# Preprocessing: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
		tenPreprocessed[:, 0, :, :] = (tenPreprocessed[:, 0, :, :] - 0.485) / 0.229
		tenPreprocessed[:, 1, :, :] = (tenPreprocessed[:, 1, :, :] - 0.456) / 0.224
		tenPreprocessed[:, 2, :, :] = (tenPreprocessed[:, 2, :, :] - 0.406) / 0.225

		# Print architecture/modules (debug console: self)
		architecture = []
		for i, m in enumerate(self.named_modules()):
			architecture.append(m)
		
		return self.moduleVgg(tenPreprocessed)
	# end
# end

class Disparity(torch.nn.Module):
	def __init__(self):
		super(Disparity, self).__init__()

		self.moduleImage = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
		self.moduleSemantics = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

		# Three 512 => semantic features
		# First three 32, 48, 64 => image information
		for intRow, intFeatures in [ (0, 32), (1, 48), (2, 64), (3, 512), (4, 512), (5, 512) ]:
			self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ])) # '0x0 - 0x1', '1x0 - 1x1', ..., '5x0 - 5x1'
			self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ])) # '0x1 - 0x2', '1x1 - 1x2', ..., '5x1 - 5x2'
			self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ])) # '0x2 - 0x3', '1x2 - 1x3', ..., '5x2 - 5x3'
		# end

		for intCol in [ 0, 1 ]:
			self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([ 32, 48, 48 ])) # '0x0 - 1x0', '0x1 - 1x1'
			self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([ 48, 64, 64 ])) # '1x0 - 2x0', '1x1 - 2x1'
			self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([ 64, 512, 512 ])) # '2x0 - 3x0', '2x1 - 3x1'
			self.add_module('3x' + str(intCol) + ' - ' + '4x' + str(intCol), Downsample([ 512, 512, 512 ])) # '3x0 - 4x0', '3x1 - 4x1'
			self.add_module('4x' + str(intCol) + ' - ' + '5x' + str(intCol), Downsample([ 512, 512, 512 ])) # '4x0 - 5x0', '4x1 - 5x1'
		# end

		for intCol in [ 2, 3 ]:
			self.add_module('5x' + str(intCol) + ' - ' + '4x' + str(intCol), Upsample([ 512, 512, 512 ])) # '5x2 - 4x2', '5x3 - 4x3'
			self.add_module('4x' + str(intCol) + ' - ' + '3x' + str(intCol), Upsample([ 512, 512, 512 ])) # '4x2 - 3x2', '4x3 - 3x3'
			self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([ 512, 64, 64 ])) # '3x2 - 2x2', '3x3 - 2x3'
			self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([ 64, 48, 48 ])) # '2x2 - 1x2', , '2x3 - 1x3'
			self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([ 48, 32, 32 ])) # '1x2 - 0x2', '1x3 - 0x3'
		# end

		self.moduleDisparity = Basic('conv-relu-conv', [ 32, 32, 1 ])
	# end

	def forward(self, tenImage, tenSemantics):
		order = []
		tenColumn = [ None, None, None, None, None, None ]

		tenColumn[0] = self.moduleImage(tenImage)
		order.append('moduleImage')
		tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0]) # Downsample ...
		order.append('0x0 - 1x0')
		tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
		order.append('1x0 - 2x0')
		order.append('<SUM>')
		tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2]) + self.moduleSemantics(tenSemantics)
		order.append('2x0 - 3x0')
		order.append('moduleSemantics')
		order.append('</SUM>')
		tenColumn[4] = self._modules['3x0 - 4x0'](tenColumn[3])
		order.append('3x0 - 4x0')
		tenColumn[5] = self._modules['4x0 - 5x0'](tenColumn[4]) # ... Downsample
		order.append('4x0 - 5x0')

		intColumn = 1
		for intRow in range(len(tenColumn)):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow]) # '0x0 - 0x1', '1x0 - 1x1', ..., '5x0 - 5x1' # Image and semantic features
			if intRow != 0:
				order.append('<SUM>')
			order.append(str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn))
			if intRow != 0: # ignore first
				tenColumn[intRow] += self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow - 1]) # '0x1 - 1x1', '1x1 - 2x1', '2x1 - 3x1', '3x1 - 4x1', '4x1 - 5x1' # Downsampling col 1
				order.append(str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn))
				order.append('</SUM>')
			# end
		# end

		intColumn = 2
		for intRow in range(len(tenColumn) -1, -1, -1):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow]) # '5x1 - 5x2', ..., '1x1 - 1x2', '0x1 - 0x2'
			if intRow != len(tenColumn) - 1:
				order.append('<SUM>')
			order.append(str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn))
			if intRow != len(tenColumn) - 1: # ignore first
				tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1]) # '5x2 - 4x2', '4x2 - 3x2', '3x2 - 2x2', '2x2 - 1x2', '1x2 - 0x2' # Upsampline col 2
				order.append(str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn))
				order.append('</SUM>')

				if tenUp.shape[2] != tenColumn[intRow].shape[2]:
					tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0) # padding
				if tenUp.shape[3] != tenColumn[intRow].shape[3]:
					tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tenColumn[intRow] += tenUp
			# end
		# end

		intColumn = 3
		for intRow in range(len(tenColumn) -1, -1, -1):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow]) # '5x2 - 5x3', ..., '1x2 - 1x3', '0x2 - 0x3'  
			if intRow != len(tenColumn) - 1:
				order.append('<SUM>')
			order.append(str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn))
			if intRow != len(tenColumn) - 1: # ignore first
				tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1]) # '5x3 - 4x3', '4x3 - 3x3', '3x3 - 2x3', '2x3 - 1x3', '1x3 - 0x3' # Upsampline col 3
				order.append(str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn))
				order.append('</SUM>')

				if tenUp.shape[2] != tenColumn[intRow].shape[2]:
					tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0) # padding
				if tenUp.shape[3] != tenColumn[intRow].shape[3]:
					tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tenColumn[intRow] += tenUp
			# end
		# end

		tr = torch.nn.functional.threshold(input=self.moduleDisparity(tenColumn[0]), threshold=0.0, value=0.0)
		order.append('<THRESHOLD>[0,0]')
		order.append('moduleDisparity')
		order.append('</THRESHOLD>')

		architecture = []
		for i, o in enumerate(order):
			if i < len(order)-1:
				if 'SUM' not in o and 'THRESHOLD' not in o:
					layersObject = self._modules[o]
				else:
					layersObject = {}
				
				architecture.append(f'name: {o}')
				architecture.append(f'layers: {layersObject}')

		return tr
	# end
# end

if torch.cuda.is_available():
	moduleSemantics = Semantics().to(device).eval()
	moduleDisparity = Disparity().to(device).eval(); moduleDisparity.load_state_dict(torch.load('./models/disparity_estimation.pytorch'))
else:
	moduleSemantics = Semantics().to(device).eval()
	moduleDisparity = Disparity().to(device).eval(); moduleDisparity.load_state_dict(torch.load('./models/disparity_estimation.pytorch', map_location=torch.device('cpu')))

def disparity_estimation(tenImage):
	# tenImage.shape = (1, 3, 768, 1024) (mini-batch, channels, height, width)
	intWidth = tenImage.shape[3]
	intHeight = tenImage.shape[2]

	fltRatio = float(intWidth) / float(intHeight)

	# Resize dimension to max 512 width or height and keep aspect ratio
	intWidth = min(int(512 * fltRatio), 512)
	intHeight = min(int(512 / fltRatio), 512)

	# Down samples the input to either the given size
	# align_corners=False:
	# the input and output tensors are aligned by the corner points of their corner pixels,
	# and the interpolation uses edge value padding for out-of-boundary values
	tenImage = torch.nn.functional.interpolate(input=tenImage, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	# 1. call Semantics().forward() 2. call Disparity().forward()
	return moduleDisparity(tenImage, moduleSemantics(tenImage))
# end
