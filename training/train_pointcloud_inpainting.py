from pointcloud_inpainting import Inpaint, init_weights
from transforms import ToTensor, RandomWarp
from dataset import ImageDepthDataset
from losses import get_kernels, derivative_scale, compute_l1_loss, compute_loss_grad, compute_loss_perception
from utils import load_model, save_model, get_eta_string, save_log, pad_number

import torch
import torchvision
from torchvision import transforms, utils
import argparse
import numpy as np
from cv2 import cv2
import os
import time
import sys

device = 'cpu'
if torch.cuda.is_available():
    import cupy
    device = torch.device('cuda')

def train(args, inpaintModel, vggModelRelu4, data_loader, optimizer, scheduler, epoch, iter_nb):

    for batch_idx, sample_batched in enumerate(data_loader):
        if batch_idx % args.log_interval == 0:
            t1 = time.time()

        image_masked, image_gt, depth_masked, depth_gt, fltFov = sample_batched['image_masked'], sample_batched['image_gt'], sample_batched['depth_masked'], sample_batched['depth_gt'], sample_batched['fltFov']

        # reset previously calculated gradients (deallocate memory)
        optimizer.zero_grad()
        
        inpaint_color, inpaint_depth = inpaintModel(image_masked, image_gt, depth_masked, depth_gt)

        # color inpainting loss cumputation
        loss_color = compute_l1_loss(inpaint_color, image_gt, device)

        with torch.no_grad(): # disable calculation of gradients
            vgg_inpaint_color, vgg_image_gt = vggModelRelu4(inpaint_color), vggModelRelu4(image_gt)

        loss_perception = compute_loss_perception(vgg_inpaint_color, vgg_image_gt, device)
        
        # depth inpainting loss computation
        loss_ord = compute_l1_loss(inpaint_depth, depth_gt, device)
        loss_grad = compute_loss_grad(inpaint_depth, depth_gt, device)
        loss_depth = 0.0001 * loss_ord + loss_grad

        # combined loss computation
        loss_inpaint = loss_color + loss_perception + loss_depth

        if len(torch.nonzero(torch.isnan(loss_inpaint.view(-1)))) > 0 or len(torch.nonzero(torch.isinf(loss_inpaint.view(-1)))) > 0:
            print('Terminate training:')
            print(f'Loss is nan or inf at iteration {(batch_idx * len(image_gt)) + (args.batch_size * args.log_interval)}')
            sys.exit()

        loss_inpaint.backward()
        torch.nn.utils.clip_grad_norm_(inpaintModel.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # print loss and progress
        if batch_idx % args.log_interval == 0:
            current_step = (batch_idx * len(image_gt)) + (args.batch_size * args.log_interval)
            total_steps = len(data_loader) * args.batch_size
            progress = 100. * current_step / total_steps

            # save output and input of model as JPEG
            file_name = f'e-{pad_number(args.epochs, epoch)}-it-{pad_number(total_steps, current_step)}-b-{args.batch_size}-l-{loss_inpaint:.8f}'
            logs_path = os.path.join(args.logs_path, file_name)
            save_log(inpaint_color, file_name=f'{logs_path}-color.jpg')
            save_log(inpaint_depth, file_name=f'{logs_path}-depth.jpg')
            save_log(image_gt, file_name=f'{logs_path}-color-gt.jpg')
            save_log(depth_gt, file_name=f'{logs_path}-depth-gt.jpg')

            # compute estimated time of arrival
            t2 = time.time()
            eta = get_eta_string(t1, t2, current_step, total_steps, epoch, args)
            print(f'Train Epoch: {epoch} [{current_step}/{total_steps} ({progress:.0f} %)]\tLoss: {loss_inpaint:.8f}\t{eta}', end='\r')
        
        # save model checkpoint every 5 % iterations
        if batch_idx % int((len(data_loader) * 0.05)) == 0:
            current_step = (batch_idx * len(image_gt)) + (args.batch_size * args.log_interval)
            total_steps = len(data_loader) * args.batch_size
            model = {
                'inpainting': {
                    'model': inpaintModel,
                    'opt': optimizer,
                    'schedule': scheduler,
                    'file_name': f'e-{pad_number(args.epochs, epoch)}-it-{pad_number(total_steps, current_step)}-b-{args.batch_size}-l-{loss_inpaint:.8f}.pt'
                }
            }
            save_model(model, iter_nb, args.models_path)
    
    iter_nb += 1
    return iter_nb


def valid(model, data_loader):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch 3d Ken Burns Effect: Disparity Refinement Training')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--valid-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=26, metavar='N',
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
                        type=str, default='../model_checkpoints_inpainting', help='Path to save model checkpoints')                
    parser.add_argument('--logs-path', action='store',
                        type=str, default='../logs_inpainting', help='Path to save logs')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                        help='Set number of workers for multiprocessing. List CPU cores with $lscpu. Disabled on Windows => num-workers=0')
    parser.add_argument('--valid-size', type=float, default=0.01, metavar='VS',
                        help='Set size of the validation dataset: e.g.: valid-size=0.01 => train-size=0.99')
    parser.add_argument('--pin-memory', action='store_true', default=False,
                        help='Speeds-up the transfer of dataset between CPU and GPU')
    parser.add_argument('--continue_training', action='store_true', default=False,
                        help='Training is continued from saved model checkpoints saved in --checkpoints-path argument)')
    return parser.parse_args()

def get_vgg19_relu4():
    vggModel = torchvision.models.vgg19_bn(pretrained=True).features
    vggModelRelu4 = torch.nn.Sequential(
        vggModel[0:3],
        vggModel[3:6],
        torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        vggModel[7:10],
        vggModel[10:13],
        torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        vggModel[14:17],
        vggModel[17:20],
        vggModel[20:23],
        vggModel[23:26],
        torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        vggModel[27:30],
        vggModel[30:33],
        vggModel[33:36],
        vggModel[36:39])
    return vggModelRelu4

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
    transform = transforms.Compose([ToTensor(device), RandomWarp()])
    dataset = ImageDepthDataset(csv_file='dataset_inpainting.csv', dataset_path=args.dataset_path, train_mode='inpainting', transform=transform)

    train_loader, valid_loader = dataset.get_train_valid_loader(
        args.batch_size,
        args.valid_batch_size,
        args.valid_size,
        args.seed,
        args.num_workers,
        args.pin_memory
    )

    iter_nb = 0

    vggModelRelu4 = get_vgg19_relu4()
    vggModelRelu4 = vggModelRelu4.to(device).eval()
    inpaintModel = Inpaint().apply(init_weights).to(device).eval()

    optimizer = torch.optim.Adam(inpaintModel.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    lambda_lr = lambda epoch: args.gamma_lr ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    # load model checkpoint for continuing training
    if args.continue_training:
        print(f'Loading model state from {str(args.models_path)}')
        model = {'model': inpaintModel, 'type': 'inpainting', 'opt': optimizer, 'schedule': scheduler}
        iter_nb = load_model(model, args.models_path, continue_training=args.continue_training)

    inpaintModel.train()

    for epoch in range(1, args.epochs + 1):
        iter_nb += train(args, inpaintModel, vggModelRelu4, train_loader, optimizer, scheduler, epoch, iter_nb)
        valid(inpaintModel, valid_loader)


if __name__ == "__main__":
    main()
