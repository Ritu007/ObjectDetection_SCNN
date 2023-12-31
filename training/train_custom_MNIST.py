from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from sklearn.metrics import confusion_matrix
from plot_confmat import plot_confusion_matrix
from Models.custom_MNIST import *
import torch
from parameters import *

# from net_model import SCNN as CONVLSTM
# from net_model_convlstm import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import cv2
import numpy as np
import random
names = 'spiking_model_custom_mnist'
data_path = './raw/'  # ta" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

print("Total Number of Samples:", len(train_dataset))
print("Total Number of Samples:", len(test_set))

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])


snn = SCNN()
# snn = CONVLSTM()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

# ================================== Train ==============================
for real_epoch in range(num_epoch):
    print("Epoch: ", real_epoch)
    running_loss = 0
    start_time = time.time()
    for epoch in range(param.sub_epoch):
        print("Sub-epoch: ", epoch)
        for i, (images, labels) in enumerate(train_loader):

            # print("Index: ", images.shape)
            # print(labels)
            # --- create zoom-in and zoom-out version of each image
            images2 = torch.empty((images.shape[0] * 2, 10, images.shape[2], images.shape[3]))
            labels2 = torch.empty((images.shape[0] * 2), dtype=torch.int64)
            for j in range(images.shape[0]):
                img0 = images[j, 0, :, :].numpy()
                rows, cols = img0.shape
                for k in range(10):
                    rand1 = random.randint(0, rows//2)
                    rand2 = random.randint(0, cols//2)
                    images2[j * 2, k, :, :] = torch.from_numpy(img0)
                    images2[j * 2, k, rand1: rand1 + rows//2, rand2: rand2 + cols//2] = 0
                    labels2[j * 2] = labels[j]
                    # print(images2[0])
                    # cv2.imshow('image', dst)
                    # cv2.waitKey(100)
                for k in range(10):
                    rand1 = random.randint(0, rows//2)
                    rand2 = random.randint(0, cols//2)
                    images2[j * 2 + 1, k, :, :] = torch.from_numpy(img0)
                    images2[j * 2 + 1, k, rand1: rand1 + rows//2, rand2: rand2 + cols//2] = 0
                    labels2[j * 2 + 1] = labels[j]

            # ----
            snn.zero_grad()
            optimizer.zero_grad()

            images2 = images2.float().to(device)
            # print(images2.shape)
            outputs = snn(images2)
            labels_ = torch.zeros(batch_size * 2, 10).scatter_(1, labels2.view(-1, 1), 1)
            print("labels2:", labels_)
            loss = criterion(outputs.cpu(), labels_)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('Real_Epoch [%d/%d], Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                        %( real_epoch, num_epoch, epoch, 5, i+1, len(train_dataset)//batch_size, running_loss))
                running_loss = 0
                print('Time elasped:', time.time() - start_time)

    # ================================== Test ==============================
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, real_epoch, learning_rate, 5)
    # cm = np.zeros((10, 10), dtype=np.int32)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images2 = torch.empty((images.shape[0] * 2, 10, images.shape[2], images.shape[3]))
            labels2 = torch.empty((images.shape[0] * 2), dtype=torch.int64)
            for j in range(images.shape[0]):
                img0 = images[j, 0, :, :].numpy()
                rows, cols = img0.shape
                theta1 = 0
                theta2 = 360
                for k in range(10):
                    rand1 = random.randint(0, rows//2)
                    rand2 = random.randint(0, cols//2)
                    images2[j * 2, k, :, :] = torch.from_numpy(img0)
                    images2[j * 2, k,  rand1: rand1 + rows//2, rand2: rand2 + cols//2] = 0
                    labels2[j * 2] = labels[j]
                    # cv2.imshow('image', dst)
                    # cv2.waitKey(100)
                for k in range(10):
                    rand1 = random.randint(0, rows//2)
                    rand2 = random.randint(0, cols//2)
                    images2[j * 2 + 1, k, :, :] = torch.from_numpy(img0)
                    images2[j * 2 + 1, k,  rand1: rand1 + rows//2, rand2: rand2 + cols//2] = 0
                    labels2[j * 2 + 1] = labels[j]
                    # print(labels2.shape)
            inputs = images2.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs)
            labels_ = torch.zeros(batch_size * 2, 10).scatter_(1, labels2.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            # print(predicted.shape)
            # ----- showing confussion matrix -----

            # cm += confusion_matrix(labels2, predicted)
            # ------ showing some of the predictions -----
            # for image, label in zip(inputs, predicted):
            #     for img0 in image.cpu().numpy():
            #         cv2.imshow('image', img0)
            #         cv2.waitKey(100)
            #     print(label.cpu().numpy())

            total += float(labels2.size(0))
            correct += float(predicted.eq(labels2).sum().item())
            if batch_idx % 100 == 0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    class_names = ['0', '1', '2', '3', '4',
             '5', '6', '7', '8', '9',]
    # plot_confusion_matrix(cm, class_names)
    print('Iters:', epoch, '\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if real_epoch % 2 == 0:
        print(acc)
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('custom_checkpoint'):
            os.mkdir('custom_checkpoint')
        torch.save(state, './custom_checkpoint/ckpt' + names + '.t7')
        best_acc = acc