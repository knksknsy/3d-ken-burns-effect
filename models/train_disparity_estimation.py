from disparity_estimation import Disparity, Semantics
from transforms import ToTensor
from dataset import ImageDepthDataset

import cupy
import torch
import torchvision
from torchvision import transforms, utils
import argparse


def get_optimizer(parameters, lr, betas):
    return torch.optim.Adam(parameters, lr=lr, betas=betas)


def loss_depth():
    pass


def train(args, model, device, data_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(data_loader):
        print(batch_idx, sample_batched['image'].size(
        ), sample_batched['depth'].size(), sample_batched['fltFov'])

        # # TODO
        # image, depth = sample_batched['image'], sample_batched['depth']
        # optimizer.zero_grad()
        # # forward pass through Semantics() network
        # output = model(image)
        # #loss = F.nll_loss(output, depth) # TODO: implement and call loss_depth()
        # #loss.backward()
        # optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(image)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        #     if args.dry_run:
        #         break


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
    transform = transforms.Compose([ToTensor()])
    dataset = ImageDepthDataset(
        csv_file='dataset.csv', dataset_path=args.dataset_path, transform=transform)
    train_loader, valid_loader = dataset.get_train_valid_loader(
        args.batch_size, args.valid_batch_size, args.valid_size, args.seed, args.num_workers, args.pin_memory)

    model = Disparity().to(device)
    optimizer = get_optimizer(
        model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        valid(model, device, valid_loader)

    if args.save_model:
        torch.save(model.state_dict(), "stdl_disparity_estimation.pt")


if __name__ == "__main__":
    main()
