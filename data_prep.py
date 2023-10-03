import torch
import cv2
import time
import numpy as np


def image_process(image_path, labels_path, input_size):
    image = cv2.imread(image_path)
    print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)
    # height, width = image.shape
    # r, s = height/64, width/64
    # image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    height, width = image.shape
    max_size = max(height, width)
    r = max_size / input_size
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    new_image = np.zeros((input_size, input_size), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized
    new_image = new_image / 255

    with open(labels_path, 'r') as fp:
        line = fp.readlines()[0].strip()

    # print(r,s)
    # print(new_image.shape)
    line = line.split(" ")
    print(line)
    label = int(line[0])
    box = np.array(line[1:], dtype=float)
    x, y, w, h = box[0], box[1], box[2], box[3]
    new_box = [int((x - 0.5 * w) * width / r), int((y - 0.5 * h) * height / r), int(w * width / r), int(h * height / r)]
    print(new_box)
    cv2.rectangle(new_image, new_box, (255, 0, 0), 2)

    return new_image, new_box, label


def create_tensor(x_input,y_input):
    # with open(labels_path, )
    X = []
    Y = []
    X.append(x_input)
    X = np.array(X)
    # X = np.expand_dims(X, axis=3)
    Y.append(y_input)
    Y = np.array(Y)
    # with tf.device("CPU"):
    #     X = tf.convert_to_tensor(X, dtype=tf.float32)
    #     Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    #
    # result = tf.data.Dataset.from_tensor_slices((X, Y))
    #
    # return result
    # X = torch.tensor(...)  # Your data
    # Y = torch.tensor(...)  # Your labels

    # Combine X and Y into a TensorDataset
    # dataset = torch.utils.data.TensorDataset(X, Y)
    x_tens = torch.tensor(X).to(torch.float32)
    y_tens = torch.tensor(Y).to(torch.float32)
    # dataset = torch.utils.data.TensorDataset(x_tens, y_tens)
    # input_tensor = tens.to(torch.float32)
    # if torch.cuda.is_available():
    #     input_tensor = input_tensor.to('cuda')
    # input_tensor = input_tensor.view(1, 64, 64)
    # input_tensor = input_tensor.permute(1, 2, 0).contiguous()
    # print(input_tensor.shape)
    return x_tens, y_tens
