from disparity_estimation import Disparity, Semantics
from transforms import ToTensor, DownscaleDepth, RandomRescaleCrop
from dataset import ImageDepthDataset
from losses import get_kernels, derivative_scale, compute_loss_ord, compute_loss_grad
from utils import load_model, save_model, get_eta_string, save_log, pad_number

import cupy
import torch
import torchvision
from torchvision import transforms, utils
import argparse
import numpy as np
from cv2 import cv2
import os
import time

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# TODO: collect metrics for plots
def train(args, disparityModel, semanticsModel, data_loader, optimizer, scheduler, epoch, iter_nb):

    for batch_idx, sample_batched in enumerate(data_loader):
        if batch_idx % args.log_interval == 0:
            t1 = time.time()

        image, depth, fltFov = sample_batched['image'], sample_batched['depth'], sample_batched['fltFov']
        #print(batch_idx, image.shape, depth.shape, fltFov.shape)

        # reset previously calculated gradients (deallocate memory)
        optimizer.zero_grad()

        # forward pass through Semantics() network
        with torch.no_grad(): # disable calculation of gradients
            semanticsOutput = semanticsModel(image)
        
        disparity = disparityModel(image, semanticsOutput)

        # reconstruction loss computation
        mask = torch.ones(depth.shape).to(device)
        loss_ord = compute_loss_ord(disparity, depth, mask)
        loss_grad = compute_loss_grad(disparity, depth, mask, device)

        # # loss weights computation
        # beta = 0.015
        # # gamma_ord = 0.03 * (1 + 2 * np.exp(-beta * iter_nb)) # for scale-invariant Loss 
        # gamma_ord = 0.001 * (1 + 200 * np.exp( - beta * iter_nb)) # for L1 loss
        # gamma_grad = 1 - np.exp(-beta * iter_nb)

        # loss_depth = gamma_ord * loss_ord + gamma_grad * loss_grad # Niklaus' paper => 0.0001 * loss_ord + loss_grad
        loss_depth = 0.0001 * loss_ord + loss_grad
        loss_depth.backward()
        torch.nn.utils.clip_grad_norm_(disparityModel.parameters(), 1)
        optimizer.step()
        scheduler.step()

        # print loss and progress
        if batch_idx % args.log_interval == 0:
            current_step = (batch_idx * len(image)) + (args.batch_size * args.log_interval)
            total_steps = len(data_loader) * args.batch_size
            progress = 100. * current_step / total_steps

            # save output and input of model as JPEG
            file_name = f'e-{pad_number(args.epochs, epoch)}-it-{pad_number(total_steps, current_step)}-b-{args.batch_size}-l-{loss_depth:.8f}'
            logs_path = os.path.join(args.logs_path, file_name)
            save_log(disparity, file_name=f'{logs_path}-disparity.jpg')
            save_log(depth, file_name=f'{logs_path}-depth.jpg')

            # compute estimated time of arrival
            t2 = time.time()
            eta = get_eta_string(t1, t2, current_step, total_steps, epoch, args)
            print(f'Train Epoch: {epoch} [{current_step}/{total_steps} ({progress:.0f} %)]\tLoss: {loss_depth:.8f}\t{eta}', end='\r')
        
        # save model checkpoint every 5 % iterations
        if batch_idx % int((len(data_loader) * 0.05)) == 0:
            current_step = (batch_idx * len(image)) + (args.batch_size * args.log_interval)
            total_steps = len(data_loader) * args.batch_size
            model = {
                'disparity': {
                    'model':disparityModel,
                    'opt':optimizer,
                    'schedule': scheduler,
                    'file_name': f'e-{pad_number(args.epochs, epoch)}-it-{pad_number(total_steps, current_step)}-b-{args.batch_size}-l-{loss_depth:.8f}.pt'
                }
            }
            save_model(model, iter_nb, args.models_path)
    
    iter_nb += 1
    return iter_nb


# TODO: Save validation metrics
def valid(model, data_loader):
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
    parser.add_argument('--gamma_lr', type=float, default=0.99999, metavar='GLR',
                        help='Sets the learning rate of each parameter group to the initial lr times the gamma_lr value.')
    parser.add_argument('--seed', type=int, default=1,
                        metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset-path', action='store',
                        type=str, help='Path to dataset')
    parser.add_argument('--models-path', action='store',
                        type=str, default='../model_checkpoints', help='Path to save model checkpoints')
    parser.add_argument('--logs-path', action='store',
                        type=str, default='../logs', help='Path to save logs')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                        help='Set number of workers for multiprocessing. List CPU cores with $lscpu. Disabled on Windows => num-workers=0')
    parser.add_argument('--valid-size', type=float, default=0.01, metavar='VS',
                        help='Set size of the validation dataset: e.g.: valid-size=0.01 => train-size=0.99')
    parser.add_argument('--pin-memory', action='store_true', default=False,
                        help='Speeds-up the transfer of dataset between CPU and GPU')
    parser.add_argument('--continue-training', action='store_true', default=False,
                        help='Training is continued from saved model checkpoints saved in --checkpoints-path argument)')
    return parser.parse_args()


def main():
    # get arguments from CLI
    args = parse_args()

    # create directory for model checkpoints
    if not os.path.exists(args.models_path):
        os.mkdir(args.models_path)

    # create directory for logs
    if not os.path.exists(args.logs_path):
        os.mkdir(args.logs_path)   

    torch.manual_seed(args.seed)

    # get train and valid dataset
    transform = transforms.Compose([DownscaleDepth(), RandomRescaleCrop(args.batch_size), ToTensor(device)])
    dataset = ImageDepthDataset(csv_file='dataset.csv', dataset_path=args.dataset_path, train_mode='estimation', transform=transform)

    train_loader, valid_loader = dataset.get_train_valid_loader(
        args.batch_size,
        args.valid_batch_size,
        args.valid_size,
        args.seed,
        args.num_workers,
        args.pin_memory
    )

    iter_nb = 0

    disparityModel = Disparity().to(device).eval()
    semanticsModel = Semantics().to(device).eval()

    optimizer = torch.optim.Adam(disparityModel.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    lambda_lr = lambda epoch: args.gamma_lr ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    # load model checkpoint for continuing training
    if args.continue_training:
        print(f'Loading model state from {str(args.models_path)}')
        model = {'model': disparityModel, 'type': 'disparity', 'opt': optimizer, 'schedule': scheduler}
        iter_nb = load_model(model, args.models_path, continue_training=args.continue_training)

    disparityModel.train()

    for epoch in range(1, args.epochs + 1):
        iter_nb += train(args, disparityModel, semanticsModel, train_loader, optimizer, scheduler, epoch, iter_nb)
        valid(disparityModel, valid_loader)


if __name__ == "__main__":
    main()
