import numpy as np
from numpy import interp
from matplotlib import pyplot as plt
import imageio
import math
# from parameters import param as par
# from recep_field import rf

timeinterval = 10

def encode2(pixels):

    #initializing spike train
    train = []

    for l in range(pixels.shape[0]):
        for m in range(pixels.shape[1]):

            temp = np.zeros([(timeinterval + 1),])

            #calculating firing rate proportional to the membrane potential
            freq = interp(pixels[l][m], [0, 255], [1,20])
            #print(pot[l][m], freq)
            # print freq

            assert freq > 0

            freq1 = math.ceil(600/freq)

            #generating spikes according to the firing rate
            k = freq1
            if(pixels[l][m]>0):
                while k<(timeinterval+1):
                    temp[k] = 1
                    k = k + freq1
            train.append(temp)
            # print sum(temp)
    return train

def encode(pot):

    #initializing spike train
    train = []

    for l in range(pot.shape[0]):
        for m in range(pot.shape[1]):

            temp = np.zeros([(timeinterval+1),])

            #calculating firing rate proportional to the membrane potential
            freq = interp(pot[l][m], [-1.069,2.781], [1,20])
            #print(pot[l][m], freq)
            # print freq
            # print(freq)
            assert freq > 0

            freq1 = math.ceil(600/freq)

            #generating spikes according to the firing rate
            k = freq1
            # print(freq1)
            if(pot[l][m]>0):
                while k<(timeinterval+1):
                    temp[int(k)] = 1
                    k = k + freq1
            train.append(temp)
            # print(temp)
    return train


def frequency_coding(image):
    rate = 1 - image/255
    spike_train = []

    for row in range(rate.shape[0]):
        for col in range(rate.shape[1]):
            spikes = np.zeros((timeinterval))
            time = rate[row][col] * (timeinterval)
            time = int(time)
            counter = 1
            new_time = time * counter
            while new_time < timeinterval:

                spikes[new_time] = 1
                counter += 1
                new_time = new_time * counter
            spike_train.append(spikes.tolist())

    return spike_train


def rate_coding(image):
    rate = 1 - image
    spike_train = []
    for i in range(timeinterval):
        spikes = np.zeros((rate.shape[0], rate.shape[1]), dtype=float)
        for row in range(rate.shape[0]):
            for col in range(rate.shape[1]):
                time = rate[row][col] * (timeinterval)
                # print(int(time))
                if i == int(time):
                    spikes[row][col] = 1
                else:
                    spikes[row][col] = 0
        spike_train.append(spikes)

    spike_train = np.array(spike_train)

    return spike_train
