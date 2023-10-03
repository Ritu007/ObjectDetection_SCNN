import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
import numpy as np
from model import SCNN

from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)




class SpikingConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpikingConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, device=device)
        self.threshold = 1.0  # Spiking threshold
        self.reset = 0.0      # Reset potential
        self.membrane_potential = torch.zeros(1, out_channels, 60, 60, device=device)  # Initialize membrane potential

    def forward(self, x):
        membrane_potential = self.membrane_potential + self.conv(x)
        spikes = torch.zeros_like(membrane_potential)
        spikes[membrane_potential >= self.threshold] = 1.0
        self.membrane_potential = membrane_potential - spikes * self.threshold + spikes * self.reset
        return spikes


# print(input)
# spikes = generate_spikes(input)
# print(spikes)
# print(tens)
# cv2.imshow("prediction", new_image)
# if cv2.waitKey(25) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
#
# time.sleep(5)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = SpikingConvLayer(1, 20, 5).to(device)
# summary(model, (1, 64, 64))





# results = scnn.forward(input)


