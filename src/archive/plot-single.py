#!/usr/bin/env python3

import argparse
import os
import numpy as np
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.style.use('bmh')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--xmax', type=float)
    parser.add_argument('--ymin', type=float, default=-100.0)
    parser.add_argument('--ymax', type=float, default=100.0)
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel('Timestep')
    plt.ylabel('Reward')

    model_path = args.model_path
    model = args.model
    t = time.time()

    trainP = os.path.join(model_path, 'train.log')
    trainData = np.loadtxt(trainP).reshape(-1, 2)
    testP = os.path.join(model_path, 'test.log')
    testData = np.loadtxt(testP).reshape(-1, 2)
    if trainData.shape[0] > 1:
        plt.plot(trainData[:,0], trainData[:,1], label="train", c="r")
    if testData.shape[0] > 1:
        testI = testData[:,0]
        testRew = testData[:,1]
        plt.plot(testI, testRew, label="test", c="b")

        N = 3
        testI_ = testI[N:]
        testRew_ = [sum(testRew[i-N:i])/N for i in range(N, len(testRew))]
        plt.plot(testI_, testRew_, label="rolling test", c="g")

    plt.ylim([args.ymin, args.ymax])
    plt.legend()
    fname = os.path.join(model_path, "reward.png")
    plt.savefig(fname, format="png", bbox_inches="tight")
    print('Created {}'.format(fname))

if __name__ == '__main__':
    main()
