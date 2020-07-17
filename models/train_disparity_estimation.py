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
import os
import time


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
        return torch.FloatTensor(np.zeros(shape=(2, tensor.shape[0]))).cuda()

    vec_first_element = torch.div(
        torch.sub(tensor[:, :, y+h, x], tensor[:, :, y, x]),
        torch.add(abs(tensor[:, :, y+h, x]), abs(tensor[:, :, y, x])))
    vec_second_element = torch.div(
        torch.sub(tensor[:, :, y, x+h], tensor[:, :, y, x]),
        torch.add(abs(tensor[:, :, y, x+h]), abs(tensor[:, :, y, x])))

    vec = torch.cat((vec_first_element.t(), vec_second_element.t()), dim=0)
    return vec


def l2_norm(tensor):
    batch_dim = range(tensor.shape[1])

    l2_norms = torch.FloatTensor(np.zeros(shape=(tensor.shape[1], 1))).cuda()

    for b in batch_dim:
        l2_norms[b] = torch.sqrt(torch.pow(tensor[0][b], 2) + torch.pow(tensor[1][b], 2))
    
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


# TODO: continue training from latest checkpoint
# TODO: collect metrics for plots
def train(args, model, semanticsModel, device, data_loader, optimizer, epoch):
    model.train()

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
        
        disparity = model(image, semanticsOutput)
        disparity = torch.nn.functional.threshold(disparity, threshold=0.0, value=0.0)

        # calculate loss
        loss = loss_depth(disparity, depth).mean()
        loss.backward()
        optimizer.step()

        # log loss and progress
        if batch_idx % args.log_interval == 0:
            current_step = (batch_idx * len(image)) + (args.batch_size * args.log_interval)
            total_steps = len(data_loader) * args.batch_size
            progress = 100. * current_step / total_steps

            # save output and input of model as JPEG
            file_name = f'{epoch}-{pad_current_step(total_steps, current_step)}-{loss:.2f}'
            logs_path = os.path.join(args.logs_path, file_name)
            save_disparity(disparity, file_name=f'{logs_path}-disparity.jpg')
            save_disparity(depth, file_name=f'{logs_path}-depth.jpg')

            t2 = time.time()
            eta = get_eta_string(t1, t2, current_step, total_steps, epoch, args)
            print(f'Train Epoch: {epoch} [{current_step}/{total_steps} ({progress:.0f} %)]\tLoss: {loss.item():.2f}\t{eta}', end='\r')
        
        # save model checkpoint every 5 % iterations
        if batch_idx % int((len(data_loader) * 0.05)) == 0:
            current_step = (batch_idx * len(image)) + (args.batch_size * args.log_interval)
            file_name = f'{epoch}-{pad_current_step(total_steps, current_step)}-{loss:.2f}'

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(args.checkpoints_path, f'{file_name}.pt'))


def get_eta_string(t1, t2, current_step, total_steps, epoch, args):
    time_per_sample = (t2 - t1) / (args.batch_size * args.log_interval)
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


def pad_current_step(max_steps, current_step):
    max_digits = len(str(max_steps))
    current_digits = len(str(current_step))

    pads_count = max_digits - current_digits

    padded_step = ''
    for d in range(pads_count):
        padded_step += '0'
    padded_step += str(current_step)
    return padded_step


# TODO: Save validation metrics
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
    parser.add_argument('--checkpoints-path', action='store',
                        type=str, default='../model_checkpoints', help='Path to save model checkpoints')
    parser.add_argument('--logs-path', action='store',
                        type=str, default='../logs', help='Path to save logs')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                        help='Set number of workers for multiprocessing. List CPU cores with $lscpu. Disabled on Windows => num-workers=0')
    parser.add_argument('--valid-size', type=float, default=0.01, metavar='VS',
                        help='Set size of the validation dataset: e.g.: valid-size=0.01 => train-size=0.99')
    parser.add_argument('--pin-memory', action='store_true', default=False,
                        help='Speeds-up the transfer of dataset between CPU and GPU')
    return parser.parse_args()


def main():
    # get arguments from CLI
    args = parse_args()

    # create directory for model checkpoints
    if not os.path.exists(args.checkpoints_path):
        os.mkdir(args.checkpoints_path)

    # create directory for logs
    if not os.path.exists(args.logs_path):
        os.mkdir(args.logs_path)

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # get train and valid dataset
    transform = transforms.Compose([DownscaleDepth(), ToTensor()])
    dataset = ImageDepthDataset(csv_file='dataset.csv', dataset_path=args.dataset_path, transform=transform)

    train_loader, valid_loader = dataset.get_train_valid_loader(
        args.batch_size,
        args.valid_batch_size,
        args.valid_size,
        args.seed,
        args.num_workers,
        args.pin_memory
    )

    model = Disparity().to(device)
    semanticsModel = Semantics().to(device).eval()

    optimizer = get_optimizer(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(1, args.epochs + 1):
        train(args, model, semanticsModel, device, train_loader, optimizer, epoch)
        valid(model, device, valid_loader)

    if args.save_model:
        torch.save(model.state_dict(), "stdl_disparity_estimation.pt")


if __name__ == "__main__":
    main()
