from disparity_estimation import Disparity, Semantics
from transforms import ToTensor, DownscaleDepth
from dataset import ImageDepthDataset

import cupy
import torch
import torchvision
from torchvision import transforms, utils
import argparse
import numpy as np
from cv2 import cv2


def get_optimizer(parameters, lr, betas):
    return torch.optim.Adam(parameters, lr=lr, betas=betas)


def loss_ord(output, target):
    loss = torch.FloatTensor(np.zeros(shape=(output.shape[0], 1))).cuda()
    x_dim = range(output.size()[2])
    y_dim = range(output.size()[1])

    for y in y_dim:
        for x in x_dim:
            loss += torch.abs(torch.sub(output[:, :, y, x], target[:, :, y, x]))

    return loss


def gh(tensor, h, y, x):
    # ignore NaN values
    x_dim = tensor.shape[3]
    y_dim = tensor.shape[2]
    if (y + h >= y_dim) or (x + h >= x_dim):
        return torch.FloatTensor(np.zeros(shape=(tensor.shape[0], 2))).cuda()

    vec_first_element = torch.div(
        # y+h > tensor.shape[1] => y
        torch.sub(tensor[:, :, y+h, x], tensor[:, :, y, x]),
        torch.add(abs(tensor[:, :, y+h, x]), abs(tensor[:, :, y, x])))
    vec_second_element = torch.div(
        torch.sub(tensor[:, :, y, x+h], tensor[:, :, y, x]),
        torch.add(abs(tensor[:, :, y, x+h]), abs(tensor[:, :, y, x])))

    return torch.cat((vec_first_element.t(), vec_second_element.t()), dim=0)


def l2_norm(tensor):
    batch_dim = range(tensor.shape[0])
    value_dim = range(tensor.shape[1])

    l2_norms = torch.FloatTensor(np.zeros(shape=(tensor.shape[0], 1))).cuda()

    for b in batch_dim:
        l2_norms[b][0] = torch.sqrt(torch.pow(tensor[b][0], 2) + torch.pow(tensor[b][1], 2))
    
    return l2_norms


def loss_grad(output, target):
    h_values = [pow(2, i) for i in range(0, 5)]
    loss = torch.FloatTensor(np.zeros(shape=(output.shape[0], 1))).cuda()
    x_dim = range(output.shape[2])
    y_dim = range(output.shape[1])

    for h in h_values:
        for y in y_dim:
            for x in x_dim:
                loss += l2_norm(
                    torch.sub(
                        gh(output, h, y, x),
                        gh(target, h, y, x)
                    )
                )

    return loss


def loss_depth(output, target):
    return torch.add(
        torch.mul(0.0001, loss_ord(output, target)),
        loss_grad(output, target)
    )


def get_inverse_depth(image, disparity, fltFov, baseline=20):
    # calculate focal length with formula: F = A/tan(a)
    max_dim = max(image.shape[2], image.shape[3]) / 2
    fltFov = fltFov / 2
    focal = max_dim / np.tan(fltFov)

    # calculate inverse depth with formula: (focal * baseline) / (0.0000001 + disparity)
    for i in range(disparity.shape[0]):
        disparity[i,:,:,:] = (focal[i] * baseline) / (0.0000001 + disparity[i,:,:,:])

    return disparity


def train(args, model, semanticsModel, device, data_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(data_loader):
        # print(batch_idx, sample_batched['image'].size(
        # ), sample_batched['depth'].size(), sample_batched['fltFov'])

        image, depth, fltFov = sample_batched['image'], sample_batched['depth'], sample_batched['fltFov']
        optimizer.zero_grad()

        # forward pass through Semantics() network
        semanticsOutput = semanticsModel(image)

        disparity = model(image, semanticsOutput)

        # Debuggin
        #cv2.imshow('Test', image[0,:,:,:].detach().cpu().numpy().transpose(1,2,0))
        #cv2.waitKey()
        #cv2.imshow('Test', disparity[0,0,:,:].detach().cpu().numpy())
        #cv2.waitKey()

        # get inverse depth of disparity
        inverse_depth = get_inverse_depth(image, disparity, fltFov)

        # TODO: save inverse_depth
        # TODO: save checkpoint every 5 %
        # TODO: resume training from checkpoint
        # TODO: collect accuracy and loss for plots

        loss = loss_depth(inverse_depth, depth).mean()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(image)}/{len(data_loader)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            cv2.imwrite(f'{epoch}-{loss}.png')


def valid(model, device, data_loader):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch 3d Ken Burns Effect: Disparity Estimation Training')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--valid-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--b1', type=float, default=0.9, metavar='B1',
                        help='beta 1 for adam optimizer (default: 0.9)')
    parser.add_argument('--b2', type=float, default=0.999, metavar='B2',
                        help='beta 1 for adam optimizer (default: 0.999)')
    parser.add_argument('--seed', type=int, default=1,
                        metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        default=False, help='For Saving the current Model')
    parser.add_argument('--dataset-path', action='store',
                        type=str, help='Path to dataset')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                        help='Set number of workers for multiprocessing. List CPU cores with $lscpu. Disabled on Windows => num-workers=0')
    parser.add_argument('--valid-size', type=float, default=0.2, metavar='VS',
                        help='Set size of the validation dataset: e.g.: valid-size=0.2 => train-size=0.8')
    parser.add_argument('--pin-memory', action='store_true', default=False,
                        help='Speeds-up the transfer of dataset between CPU and GPU')
    return parser.parse_args()


def main():
    # get arguments from CLI
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # get train and valid dataset
    transform = transforms.Compose([DownscaleDepth(), ToTensor()])

    dataset = ImageDepthDataset(
        csv_file='dataset.csv', dataset_path=args.dataset_path, transform=transform)
    train_loader, valid_loader = dataset.get_train_valid_loader(
        args.batch_size, args.valid_batch_size, args.valid_size, args.seed, args.num_workers, args.pin_memory)

    model = Disparity().to(device)
    semanticsModel = Semantics().to(device).eval()

    optimizer = get_optimizer(
        model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(1, args.epochs + 1):
        train(args, model, semanticsModel, device,
              train_loader, optimizer, epoch)
        valid(model, device, valid_loader)

    if args.save_model:
        torch.save(model.state_dict(), "stdl_disparity_estimation.pt")


if __name__ == "__main__":
    main()
