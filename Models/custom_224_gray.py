import torch
import torch.nn as nn
import torch.nn.functional as F
from surrogate import *
import parameters as param

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)



# define approximate firing function
act_fun = ActFun.apply


# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * param.decay + ops(x)
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike


# cnn_layer(in_planes(channels), out_planes(channels), kernel_size, stride, padding)
cfg_cnn = [(1, 32, 3, 1, 1),
           (32, 64, 3, 1, 1),
           (64, 128, 3, 2, 1), ]
# kernel size
# cnn output shapes (conv1, conv2, fc1 input)
cfg_kernel = [224, 112, 56, 28, 14]  # conv layers input image shape (+ last output shape)
# fc layer
cfg_fc = [4096, 512, param.num_classes]  # linear layers output


# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


class SpikingCNN(nn.Module):
    def __init__(self):
        super(SpikingCNN, self).__init__()
        in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, device= device)

        in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, device= device)

        in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, device= device)

        # in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[3]
        # self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, device= device)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])

    def forward(self, input, time_window=10):
        # convolutional layers membrane potential and spike memory
        c1_mem = c1_spike = torch.zeros(param.batch_size * 2, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(param.batch_size * 2, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        # linear layers membrane potential and spike memory
        c3_mem = c3_spike = torch.zeros(param.batch_size * 2, cfg_cnn[2][1], cfg_kernel[3], cfg_kernel[3], device=device)
        # c4_mem = c4_spike = torch.zeros(param.batch_size * 2, cfg_cnn[3][1], cfg_kernel[4], cfg_kernel[4], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(param.batch_size * 2, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(param.batch_size * 2, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(param.batch_size * 2, cfg_fc[2], device=device)

        for step in range(time_window):  # simulation time steps
            # print("For the time stamp:", step)
            x = input[:, step: step + 1, :, :]
            # print("The value of X is:", x)
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            # print("The value of c1 is:", c1_mem, c1_spike)
            x = F.avg_pool2d(c1_spike, 2, stride=2, padding=0)

            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem, c2_spike)
            # print("The value of c2 is:", c2_mem, c2_spike)
            x = F.avg_pool2d(c2_spike, 2, stride=2, padding=0)

            c3_mem, c3_spike = mem_update(self.conv3, x, c3_mem, c3_spike)
            # print("The value of c3 is:", c3_mem, c3_spike)
            x = F.avg_pool2d(c3_spike, 2, stride=1, padding=0)

            # c4_mem, c4_spike = mem_update(self.conv4, x, c4_mem, c4_spike)
            # # print("The value of c3 is:", c3_mem, c3_spike)
            # x = F.avg_pool2d(c4_spike, 2, stride=2, padding=0)

            x = x.view(param.batch_size * 2, -1)  # flatten

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            # print("The value of h1 is:", h1_mem, h1_spike)
            h1_sumspike += h1_spike

            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            # print("The value of h2 is:", h2_mem, h2_spike)
            h2_sumspike += h2_spike

            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            # print("The value of h2 is:", h2_mem, h2_spike)
            h3_sumspike += h3_spike

        outputs = h3_sumspike / time_window
        return outputs