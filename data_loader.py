import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import cv2
import numpy as np


input_size = 28

class ObjectDetectionDataset(Dataset):

    def __init__(self, img_folder, annotation_folder, transform = None):
        self.img_folder = img_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.img_paths = [os.path.join(img_folder, img) for img in os.listdir(img_folder)]
        self.annotation_path = [os.path.join(annotation_folder, annot) for annot in os.listdir(annotation_folder)]


    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        annotation_path = self.annotation_path[idx]

        # img = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        with open(annotation_path, 'r') as fp:
            line = fp.readlines()[0].strip()

        values = line.split(" ")
        # print(values)
        box = np.array(values[1:], dtype=float)
        label = int(line[0])
        # bbox_info = np.array(values, dtype=float)
        # print(bbox_info)
        boxes = torch.tensor(box, dtype=torch.float32)
        labels = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)

        return img, labels


def transform(img, box, label):
    height, width = img.shape
    max_size = max(height, width)
    r = max_size / input_size
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    new_image = np.zeros((input_size, input_size), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized

    x, y, w, h = box[0], box[1], box[2], box[3]
    new_box = [int((x - 0.5 * w) * width / r), int((y - 0.5 * h) * height / r), int(w * width / r), int(h * height / r)]

    img = ToTensor()(new_image)

    return img, new_box, label


