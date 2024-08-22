from datetime import datetime
import time
import csv

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from scipy.integrate import odeint

import occamnet.Bases as Bases
from occamnet.Losses import CrossEntropyLoss
from occamnet.Network import NetworkConstants
from occamnet.SparseSetters import SetNoSparse as SNS


def model(data, t):
    k, y, s, n, delta, g_b, alpha = data
    dydt = [s*y-k*(delta+g_b+n), alpha*y/k*(s*y-k*(delta+n+g_b)), 0.05, 0.05*n, 0, 0, 0]
    return dydt

if __name__ == '__main__':

    ################ Generate data ################

    size = 20
    s = np.random.uniform(low=0.2, high=0.8, size=(size,)) 
    n = np.random.uniform(low=0.05, high=0.1, size=(size,))  
    delta = np.random.uniform(low=0.05, high=0.1, size=(size,))
    g_b = np.random.uniform(low=0.05, high=0.1, size=(size,))
    alpha = np.random.uniform(low=0.01, high=0.5, size=(size,))
    k0 = np.random.uniform(low=0.01, high=0.1, size=(size,))
    y0 = k0**alpha

    t = np.arange(30) # time points

    # solve ODE
    model_fit = [odeint(model, [k0[i], y0[i], s[i], delta[i], n[i], g_b[i], alpha[i]], t) for i in range(len(s))]
    X = [torch.FloatTensor(xi)[:, [0, 1, 2, 3]] for xi in model_fit]

    # Fit dk/dt (index 0)
    Y = [torch.FloatTensor(np.diff(X[i][:, [0]], axis=0)[1:] - 0.5*np.diff(np.diff(X[i][:, [0]], axis=0), axis=0)) for i in range(len(X))]
    X = [X[i][1:-1, :] for i in range(len(X))]
  
    inputSize = 4 # Number of input variables in each individual dataset
    outputSize = 1 # Number of output variables in each individual dataset

    units = [np.array([1, -1]),np.array([1, -1]), np.array([0, 0]), np.array([0, 0]), np.array([1, -1])]


    ################ Initialize OccamNet ################

    ensembleMode = True # Toggle ensemble learning

    # Default hyperparameters
    epochs = 2
    batchesPerEpoch = 1
    learningRate = 1
    constantLearningRate = 0.05
    decay = 1
    temp = 10
    endTemp = 10
    sampleSize = 1000 # Number of functions to sample

    # Regularization parameters
    activationWeight = 0
    constantWeight = 0

    # Sweep parameters
    sDev_sweep = [0.5]
    top_sweep = [5]
    equalization_sweep = [1]

    # Activation layers
    layers = [[Bases.Add(), Bases.Subtract(), Bases.Multiply(),  Bases.Divide(), Bases.AddConstant(), Bases.MultiplyConstant(), Bases.Square(), Bases.PowerConstant(), Bases.Exp(), Bases.Log(), Bases.Sin(), Bases.Cos()],
            [Bases.Add(), Bases.Subtract(), Bases.Multiply(), Bases.Divide(), Bases.AddConstant(), Bases.MultiplyConstant(), Bases.Square(), Bases.PowerConstant(), Bases.Exp(), Bases.Log(), Bases.Sin(), Bases.Cos()],
            [Bases.Add(), Bases.Subtract(), Bases.Multiply(), Bases.Divide(), Bases.AddConstant(), Bases.MultiplyConstant(), Bases.Square(), Bases.PowerConstant(), Bases.Exp(), Bases.Log(), Bases.Sin(), Bases.Cos()]]

    
    ################ Training ################

    file_name = "SolowDemo"
    date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3] 
    file_path = 'results/' + file_name + '_' + date_time + ".csv"

    with open(file_path, 'a') as f:
        writer = csv.writer(f)

        header = ['wmse', 'mse_sDev', 'mse_median', 'expression', 'sDev', 'top', 'equalization', 'runtime']

        writer.writerow(header)

    for sDev in sDev_sweep:
        for top in top_sweep:
            for equalization in equalization_sweep:

                print('Training with parameters: sDev={sDev}, top={top}, equalization={eq}'.format(
                    sDev=sDev, 
                    top=top,
                    eq=equalization))

                start = time.time()

                loss = CrossEntropyLoss(sDev, 
                                        top, 
                                        anomWeight=0, 
                                        constantWeight=constantWeight, 
                                        activationWeight=activationWeight)

                sparsifier = SNS()

                n = NetworkConstants(inputSize, 
                                    layers, 
                                    outputSize, 
                                    sparsifier, 
                                    loss, 
                                    learningRate, 
                                    constantLearningRate, 
                                    temp, 
                                    endTemp, 
                                    equalization, 
                                    skipConnections = True)

                n.setConstants([0 for j in range(n.totalConstants)])

                train_function = n.trainFunction(epochs, 
                                                batchesPerEpoch, 
                                                sampleSize, 
                                                decay, 
                                                X, 
                                                Y, 
                                                useMultiprocessing = True, 
                                                numProcesses = 20, 
                                                ensemble=ensembleMode,
                                                units=units)

                ### Evaluation ###

                output = n.forwardFitConstants(train_function, X, Y, ensemble=True)
                output = output.squeeze(1)

                # Weighted MSE
                MSELoss = nn.MSELoss()
                losses = []

                for curr_Y in Y:
                    curr_out, output = torch.split(output, [curr_Y.shape[0], output.shape[0]-curr_Y.shape[0]])
                    losses.append(MSELoss(curr_Y, curr_out).item())
            
                weighted_mse = np.mean(losses)
                std_mse = np.std(losses)
                median_mse = np.median(losses)

                expression = str(n.applySymbolicConstant(train_function))

                end = time.time()
                minutes = (end - start)/60

                with open(file_path, 'a') as f:
                    writer = csv.writer(f)

                    data = [weighted_mse, std_mse, median_mse , expression, sDev, top, equalization, minutes]

                    writer.writerow(data)

           