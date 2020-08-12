import torch
import torchvision
from cv2 import cv2
import os
import time

def load_model(model, models_path, continue_training=False):
    iter_nb = 0

    # Get latest model .pt files in directory models_path
    model_path = [f for f in os.listdir(models_path) if f.endswith('.pt')][-1]
    checkpoint = torch.load(os.path.join(models_path, model_path))
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


def get_eta_string(t1, t2, current_step, total_steps, epoch, args):
    time_per_sample = (t2 - t1) / args.batch_size
    estimated_time_arrival = ((total_steps * args.epochs) - (current_step + ((epoch - 1) * total_steps))) * time_per_sample
    left_epochs = args.epochs + 1 - epoch

    tps = f'{time_per_sample:.2f} s/sample'
    eta_d = estimated_time_arrival // (60 * 60 * 24)
    eta_hms = time.strftime('%Hh %Mm %Ss', time.gmtime(int(estimated_time_arrival)))

    return f'{tps}\tETA ({left_epochs} Epochs): {eta_d:.0f}d {eta_hms}'


def save_log(tensor, file_name):
    if tensor.shape[1] == 1: # log disparity image
        tensor_out = (tensor[0,0,:,:] / 20 * 255.0).clamp(0.0, 255.0).type(torch.uint8)
        tensor_out = tensor_out.detach().cpu().numpy()
        cv2.imwrite(file_name, tensor_out)
    elif tensor.shape[1] == 3: # log color image
        tensor_out = (tensor[0,:,:,:] * 255.0).permute(1,2,0).clamp(0.0, 255.0).type(torch.uint8)
        tensor_out = tensor_out.detach().cpu().numpy()
        cv2.imwrite(file_name, tensor_out)


def pad_number(max_number, current_number):
    max_digits = len(str(max_number))
    current_digits = len(str(current_number))
    pads_count = max_digits - current_digits

    padded_step = ''
    for d in range(pads_count):
        padded_step += '0'
    padded_step += str(current_number)
    return padded_step