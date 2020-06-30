import torch
import torchvision
import cupy
from disparity_estimation import Disparity
from zipfile import ZipFile
from PIL import Image
from torchvision.transforms import ToTensor
import os
import numpy as np
from cv2 import cv2

def optimizer(parameters):
    return torch.optim.Adam(parameters, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

def loss_depth():
    pass

def train(model, data_paths, optimizer, epoch):
    model.train()
    for batch_idx, (image_zip_path, target_zip_path) in enumerate(data_paths):
        archive_image = ZipFile(image_zip_path, 'r')
        archive_target = ZipFile(target_zip_path, 'r')

        image_paths = [d for d in archive_image.namelist() if d.endswith('.png')]
        target_paths = [d for d in archive_target.namelist() if d.endswith('.exr')]

        for image_path, target_path in zip(image_paths, target_paths):
            image = archive_image.read(image_path)
            imageTensor = torch.from_numpy(cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)).cuda()
            target = archive_target.read(target_path)
            targetTensor = torch.from_numpy(cv2.imdecode(np.frombuffer(target, np.uint8), cv2.IMREAD_ANYDEPTH)).cuda()

            optimizer.zero_grad()
            output = model(imageTensor)

            # loss = F.nll_loss(output, target)
            # loss.backward()

            # optimizer.step()
            # if batch_idx % args.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item()))

def data_loader():
    data_path = 'D:/3d-ken-burns-dataset'
    image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if '-depth' not in f and f.endswith('.zip')]
    label_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if '-depth' in f and f.endswith('.zip')]

    return zip(image_paths, label_paths)

# total_image_paths = []

# archive = ZipFile(self.data_path, 'r')

# img_path="/data/v_PlayingSitar_g25_c04_0256.jpeg"
# image = Image.open(img_path)
# image = ToTensor()(image).unsqueeze(0) # unsqueeze to add artificial first dimension
# image = Variable(image)

if __name__ == "__main__":
    kwargs = {'num_workers': 1, 'pin_memory': True}

    model = Disparity().cuda()
    data_paths = data_loader()

    epochs = int(3*10e6)
    optimizer = optimizer(model.parameters())

    for epoch in range(1, epochs +1):
        train(model, data_paths, optimizer, epoch)
