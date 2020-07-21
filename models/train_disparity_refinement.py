from disparity_refinement import Refine
from disparity_estimation import Disparity, Semantics
from transforms import ToTensor
from dataset import ImageDepthDataset

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

def load_model(model, models_path, continue_training=False):
    iter_nb = 0

    # Get latest model .pt files in directory models_path
    model_path = [f for f in os.listdir(models_path) if f.endswith('.pt')][-1]
    checkpoint = torch.load(model_path)
    model['model'].load_state_dict(checkpoint['model_state_dict'])

    if continue_training:
        model['opt'].load_state_dict(checkpoint[f'optimizer_{model["type"]}_state_dict'])
        model['schedule'].load_state_dict(checkpoint[f'scheduler_{model["type"]}_state_dict'])
        iter_nb = checkpoint['iter_nb']
        print(f'Model {model["type"]} loaded succesfully.')

    return iter_nb


def save_model(models_dict, iter_nb, path):
    for model_type, model in models_dict.items():
        torch.save({
            'iter_nb': iter_nb,
            'model_state_dict': model['model'].state_dict(),
            f'optimizer_{model_type}_state_dict': model['opt'].state_dict(),
            f'scheduler_{model_type}_state_dict': model['schedule'].state_dict()},
            os.path.join(path, model['file_name']))


def get_optimizer(parameters, lr, betas):
    return torch.optim.Adam(parameters, lr=lr, betas=betas)


# Create kernels used for gradient computation
def get_kernels(h):
    kernel_elements = [-1] + [0 for _ in range(h-1)] + [1]
    kernel_elements_div = [1] + [0 for _ in range(h-1)] + [1]
    weight_y = torch.Tensor(kernel_elements).view(1,-1)
    weight_x = weight_y.T
    
    weight_y_norm = torch.Tensor(kernel_elements_div).view(1,-1)
    weight_x_norm = weight_y_norm.T
    
    return weight_x.to(device), weight_x_norm.to(device), weight_y.to(device), weight_y_norm.to(device)


def derivative_scale(tensor, h, norm=True):
    kernel_x, kernel_x_norm, kernel_y, kernel_y_norm = get_kernels(h)
    
    diff_x = torch.nn.functional.conv2d(tensor, kernel_x.view(1,1,-1,1))
    diff_y = torch.nn.functional.conv2d(tensor, kernel_y.view(1,1,1,-1))

    if norm:
        norm_x = torch.nn.functional.conv2d(torch.abs(tensor), kernel_x_norm.view(1,1,-1,1))
        norm_y = torch.nn.functional.conv2d(torch.abs(tensor), kernel_y_norm.view(1,1,1,-1))
        diff_x = diff_x/(norm_x + 1e-7)
        diff_y = diff_y/(norm_y + 1e-7)
    
    return torch.nn.functional.pad(diff_x, (0, 0, h, 0)), torch.nn.functional.pad(diff_y, (h, 0, 0, 0))


def compute_loss_ord(disparity, target, mask):
    L1Loss = torch.nn.L1Loss(reduction='sum')

    loss = 0
    N = torch.sum(mask)
    
    if N != 0:
        loss = L1Loss(disparity * mask, target * mask) / N
    return loss


def compute_loss_grad(disparity, target, mask):        
    scales = [2**i for i in range(4)]
    MSELoss = torch.nn.MSELoss(reduction='sum')

    loss = 0
    for h in scales:
        g_h_disparity_x, g_h_disparity_y = derivative_scale(disparity, h, norm=True)
        g_h_target_x, g_h_target_y = derivative_scale(target, h, norm=True)
        N = torch.sum(mask)

        if N != 0:
            loss += MSELoss(g_h_disparity_x * mask, g_h_target_x * mask) / N
            loss += MSELoss(g_h_disparity_y * mask, g_h_target_y * mask) / N

    return loss


# TODO: collect metrics for plots
def train(args, refinementModel, disparityModel, semanticsModel, data_loader, optimizer, scheduler, epoch, iter_nb):

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
            estimated_disparity = disparityModel(image, semanticsOutput)
        
        refined_disparity = refinementModel(image, estimated_disparity)
        refined_disparity = torch.nn.functional.threshold(refined_disparity, threshold=0.0, value=0.0)

        # reconstruction loss computation
        mask = torch.ones(depth.shape).to(device)
        loss_ord = compute_loss_ord(refined_disparity, depth, mask)
        loss_grad = compute_loss_grad(refined_disparity, depth, mask)

        loss_depth = 0.0001 * loss_ord + loss_grad
        loss_depth.backward()
        torch.nn.utils.clip_grad_norm_(refinementModel.parameters(), 1)
        optimizer.step()
        scheduler.step()

        # print loss and progress
        if batch_idx % args.log_interval == 0:
            current_step = (batch_idx * len(image)) + (args.batch_size * args.log_interval)
            total_steps = len(data_loader) * args.batch_size
            progress = 100. * current_step / total_steps

            # save output and input of model as JPEG
            file_name = f'e-{pad_number(args.epochs, epoch)}-it-{pad_number(total_steps, current_step)}-b-{args.batch_size}-l-{loss_depth:.2f}'
            logs_path = os.path.join(args.logs_path, file_name)
            save_disparity(refined_disparity, file_name=f'{logs_path}-disparity.jpg')
            save_disparity(depth, file_name=f'{logs_path}-depth.jpg')

            # compute estimated time of arrival
            t2 = time.time()
            eta = get_eta_string(t1, t2, current_step, total_steps, epoch, args)
            print(f'Train Epoch: {epoch} [{current_step}/{total_steps} ({progress:.0f} %)]\tLoss: {loss_depth:.2f}\t{eta}', end='\r')
        
        # save model checkpoint every 5 % iterations
        if batch_idx % int((len(data_loader) * 0.05)) == 0:
            current_step = (batch_idx * len(image)) + (args.batch_size * args.log_interval)
            total_steps = len(data_loader) * args.batch_size
            model = {
                'refinement': {
                    'model':refinementModel,
                    'opt':optimizer,
                    'schedule': scheduler,
                    'file_name': f'e-{pad_number(args.epochs, epoch)}-it-{pad_number(total_steps, current_step)}-b-{args.batch_size}-l-{loss_depth:.2f}.pt'
                }
            }
            save_model(model, iter_nb, args.models_path)
    
    iter_nb += 1
    return iter_nb


def get_eta_string(t1, t2, current_step, total_steps, epoch, args):
    time_per_sample = (t2 - t1) / args.batch_size
    estimated_time_arrival = ((total_steps * args.epochs) - (current_step + ((epoch - 1) * total_steps))) * time_per_sample
    left_epochs = args.epochs + 1 - epoch

    tps = f'{time_per_sample:.2f} s/sample'
    eta_d = estimated_time_arrival // (60 * 60 * 24)
    eta_hms = time.strftime('%Hh %Mm %Ss', time.gmtime(int(estimated_time_arrival)))

    return f'{tps}\tETA ({left_epochs} Epochs): {eta_d:.0f}d {eta_hms}'

def save_disparity(disparity, file_name):
    disparity_out = (disparity[0,0,:,:] / 20 * 255.0).clamp(0.0, 255.0).type(torch.uint8)
    disparity_out = disparity_out.detach().cpu().numpy()
    cv2.imwrite(file_name, disparity_out)


def pad_number(max_number, current_number):
    max_digits = len(str(max_number))
    current_digits = len(str(current_number))
    pads_count = max_digits - current_digits

    padded_step = ''
    for d in range(pads_count):
        padded_step += '0'
    padded_step += str(current_number)
    return padded_step


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
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
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
    parser.add_argument('--save-model', action='store_true',
                        default=False, help='For Saving the current Model')
    parser.add_argument('--dataset-path', action='store',
                        type=str, help='Path to dataset')
    parser.add_argument('--models-path', action='store',
                        type=str, default='../model_checkpoints', help='Path to save model checkpoints')                
    parser.add_argument('--disparity-path', action='store',
                        type=str, help='Path to trained disparity model')
    parser.add_argument('--logs-path', action='store',
                        type=str, default='../logs', help='Path to save logs')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                        help='Set number of workers for multiprocessing. List CPU cores with $lscpu. Disabled on Windows => num-workers=0')
    parser.add_argument('--valid-size', type=float, default=0.01, metavar='VS',
                        help='Set size of the validation dataset: e.g.: valid-size=0.01 => train-size=0.99')
    parser.add_argument('--pin-memory', action='store_true', default=False,
                        help='Speeds-up the transfer of dataset between CPU and GPU')
    parser.add_argument('--continue_training', action='store_true', default=False,
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
    transform = transforms.Compose([ToTensor()])
    dataset = ImageDepthDataset(csv_file='dataset.csv', dataset_path=args.dataset_path, transform=transform)

    train_loader, valid_loader = dataset.get_train_valid_loader(
        args.batch_size,
        args.valid_batch_size,
        args.valid_size,
        args.seed,
        args.num_workers,
        args.pin_memory
    )

    iter_nb = 0

    disparityModel = Disparity().to(device).eval(); disparityModel.load_state_dict(torch.load(args.disparity_path))
    semanticsModel = Semantics().to(device).eval()
    refinementModel = Refine().to(device).eval()

    optimizer = get_optimizer(refinementModel.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    lambda_lr = lambda epoch: args.gamma_lr ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    # load model checkpoint for continuing training
    if args.continue_training:
        print(f'Loading model state from {str(args.models_path)}')
        model = {'model': disparityModel, 'type': 'disparity', 'opt': optimizer, 'schedule': scheduler}
        iter_nb = load_model(model, args.models_path, continue_training=args.continue_training)

    refinementModel.train()

    for epoch in range(1, args.epochs + 1):
        iter_nb += train(args, refinementModel, disparityModel, semanticsModel, train_loader, optimizer, scheduler, epoch, iter_nb)
        valid(refinementModel, valid_loader)

    if args.save_model:
        torch.save(refinementModel.state_dict(), "stdl_disparity_estimation.pt")


if __name__ == "__main__":
    main()
