# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from data_prep import image_process, create_tensor
from model import SCNN
from spiking_conv import SpikingConvLayer
from encoding import rate_coding
import numpy as np
import torch
from new_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = "E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/train/images/Abyssinian_1_jpg.rf.d73587ecddc57fbabd4ee18198322975.jpg"
labels_path = "E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/train/labels/Abyssinian_1_jpg.rf.d73587ecddc57fbabd4ee18198322975.txt"
input_size = 28

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def generate_spikes(input):
    spike_conv = SpikingConvLayer(1, 20, 5)
    spikes = spike_conv.forward(input)
    print(spikes.shape)
    return spikes

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("hell")
    image, box, label = image_process(image_path, labels_path, input_size)
    print(image.shape)
    spikes = rate_coding(image)
    box = np.asarray(box, dtype=float) / input_size
    label = np.append(box, label)

    print(label)
    x_tens, y_tens = create_tensor(spikes, label)
    # print(x_tens)

    print(x_tens.shape, y_tens.shape)
    x_tens = x_tens.to(device)
    model = SpikingCNN().to(device)
    output = model.forward(x_tens)

    print(output)
    # summary(model, (10, 28, 28))
    # print(spikes)
    # file = open("file1.txt", "w+")
    #
    # # Saving the array in a text file
    # content = str(spikes)
    # file.write(content)
    # file.close()

    # input_tensor = create_tensor(image)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
