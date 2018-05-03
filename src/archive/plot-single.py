#!/usr/bin/env python3

import argparse
import os
import numpy as np
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.style.use('bmh')
plt.style.use('seaborn-whitegrid')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--n_trial', type=int, default=1)
    parser.add_argument('--xmax', type=float)
    parser.add_argument('--ymin', type=float, default=-100.0)
    parser.add_argument('--ymax', type=float, default=100.0)
    #parser.add_argument('--ymin', type=float)
    #parser.add_argument('--ymax', type=float)
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel('Episode', fontsize=20.0)
    plt.ylabel('Reward', fontsize=20.0)

    model_path = args.model_path
    model = args.model
    n_trial = args.n_trial
    t = time.time()


    D_train_x = None
    D_train_y = None
    D_test_x = None
    D_test_y = None

    for i in range(n_trial):
        trainP = os.path.join(model_path, 'train_{}.log'.format(i))
        trainData = np.loadtxt(trainP).reshape(-1, 2)
        testP = os.path.join(model_path, 'test_{}.log'.format(i))
        testData = np.loadtxt(testP).reshape(-1, 2)

        if D_train_x is None:
            D_train_x = trainData[:,0]
            D_train_y = trainData[:,1]
            D_test_x = testData[:,0]
            D_test_y = testData[:,1]

            n_step_train = D_train_x.shape[0]
            n_step_test = D_test_x.shape[0]

        else:
            D_train_x = np.vstack((D_train_x, trainData[:n_step_train,0]))
            D_train_y = np.vstack((D_train_y, trainData[:n_step_train,1]))
            D_test_x = np.vstack((D_test_x, testData[:n_step_test,0]))
            D_test_y = np.vstack((D_test_y, testData[:n_step_test,1]))



    if D_train_x.shape[0] > 1:
        #plt.plot(trainData[:,0], trainData[:,1], label="train", c="r")
        dy = D_train_y.std(axis=0)
        #plt.errorbar(D_train_x.mean(axis=0), D_train_y.mean(axis=0), label="train", yerr=dy, fmt="o", color="r")
        plt.plot(D_train_x.mean(axis=0), D_train_y.mean(axis=0), label="train", color="r")

    if D_test_x.shape[0] > 1:
        #testI = testData[:,0]
        #testRew = testData[:,1]
        #plt.plot(testI, testRew, label="test", c="b")
        #dy = D_test_y.std(axis=0)
        #plt.errorbar(D_test_x.mean(axis=0), D_test_y.mean(axis=0), label="test", yerr=dy, fmt="o", color="b")

        plt.plot(D_test_x.mean(axis=0), D_test_y.mean(axis=0), label="train", marker="x",
                linestyle="--", color="b")

        #N = 3
        #testI = D_test_x.mean(axis=0)
        #testRew = D_test_y.mean(axis=0)
        #testI_ = testI[N:]
        #testRew_ = [sum(testRew[i-N:i])/N for i in range(N, len(testRew))]
        #plt.plot(testI_, testRew_, label="rolling test", c="g", alpha=0.3)

    plt.ylim([args.ymin, args.ymax])
    plt.legend(fontsize=20.0)
    fname = os.path.join(model_path, "reward_{}.png".format(model))
    plt.savefig(fname, format="png", bbox_inches="tight")
    print('Created {}'.format(fname))

if __name__ == '__main__':
    main()
