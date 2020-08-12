import torch
import torchvision

# Create kernels used for gradient computation
def get_kernels(h, device):
    kernel_elements = [-1] + [0 for _ in range(h-1)] + [1]
    kernel_elements_div = [1] + [0 for _ in range(h-1)] + [1]
    weight_y = torch.Tensor(kernel_elements).view(1,-1)
    weight_x = weight_y.T
    
    weight_y_norm = torch.Tensor(kernel_elements_div).view(1,-1)
    weight_x_norm = weight_y_norm.T
    
    return weight_x.to(device), weight_x_norm.to(device), weight_y.to(device), weight_y_norm.to(device)


def derivative_scale(tensor, h, device, norm=True):
    kernel_x, kernel_x_norm, kernel_y, kernel_y_norm = get_kernels(h, device)
    
    diff_x = torch.nn.functional.conv2d(tensor, kernel_x.view(1,1,-1,1))
    diff_y = torch.nn.functional.conv2d(tensor, kernel_y.view(1,1,1,-1))

    if norm:
        norm_x = torch.nn.functional.conv2d(torch.abs(tensor), kernel_x_norm.view(1,1,-1,1))
        norm_y = torch.nn.functional.conv2d(torch.abs(tensor), kernel_y_norm.view(1,1,1,-1))
        diff_x = diff_x/(norm_x + 1e-7)
        diff_y = diff_y/(norm_y + 1e-7)
    
    return torch.nn.functional.pad(diff_x, (0, 0, h, 0)), torch.nn.functional.pad(diff_y, (h, 0, 0, 0))


def compute_l1_loss(output, target):
    L1Loss = torch.nn.L1Loss(reduction='mean')
    loss = L1Loss(output, target)

    if torch.isnan(loss) or torch.isinf(loss):
        loss = 0
        print(iteration)

    return loss


def compute_loss_grad(output, target, device):        
    scales = [2**i for i in range(4)]
    MSELoss = torch.nn.MSELoss(reduction='mean')

    loss = 0
    for h in scales:
        g_h_disparity_x, g_h_disparity_y = derivative_scale(output, h, device, norm=True)
        g_h_target_x, g_h_target_y = derivative_scale(target, h, device, norm=True)

        loss += MSELoss(g_h_disparity_x, g_h_target_x)
        loss += MSELoss(g_h_disparity_y, g_h_target_y)

    return loss


def compute_loss_perception(output, target):
    MSELoss = torch.nn.MSELoss(reduction='mean')
    loss = MSELoss(output, target)

    if torch.isnan(loss) or torch.isinf(loss):
        loss = 0

    return loss
