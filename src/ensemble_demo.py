from datetime import datetime
import time
import csv

import numpy as np
import torch
import torch.nn as nn

import occamnet.Bases as Bases
from occamnet.Losses import CrossEntropyLoss
from occamnet.Network import NetworkConstants
from occamnet.SparseSetters import SetNoSparse as SNS


def func(x, a, b):
    return a * np.power(x, b)


if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    ################ Generate data ################

    # Generate constant values for 5 panels
    a_vals = [2, 4, 6, 8, 10]
    b_vals = [0.1, 0.2, 0.3, 0.4, 0.5]

    X = []
    Y = []

    for a, b in zip(a_vals, b_vals):
        # Individual panels can have varying sizes
        panel_size = np.random.randint(low=50, high=150)
        x = np.random.uniform(low=0.1, high=1.0, size=(panel_size, 1))

        y = func(x, a, b)

        X.append(torch.FloatTensor(x))
        Y.append(torch.FloatTensor(y))

    inputSize = 1  # Number of input variables in each individual dataset
    outputSize = 1  # Number of output variables in each individual dataset

    ################ Initialize OccamNet ################

    ensembleMode = True  # Toggle ensemble learning

    # Hyperparameters
    epochs = 100
    batchesPerEpoch = 1
    learningRate = 1
    constantLearningRate = 0.05
    decay = 1
    temp = 10
    endTemp = 10
    sampleSize = 100  # Number of functions to sample

    # Regularization parameters
    activationWeight = 0
    constantWeight = 0

    # Sweep parameters
    sDev_sweep = [0.5, 5, 50]
    top_sweep = [1, 5, 10]
    equalization_sweep = [0, 1, 5]

    # Activation layers
    layers = [
        [
            Bases.Add(),
            Bases.Subtract(),
            Bases.Multiply(),
            Bases.Divide(),
            Bases.AddConstant(),
            Bases.MultiplyConstant(),
            Bases.Square(),
            Bases.PowerConstant(),
        ],
        [
            Bases.Add(),
            Bases.Subtract(),
            Bases.Multiply(),
            Bases.Divide(),
            Bases.AddConstant(),
            Bases.MultiplyConstant(),
            Bases.Square(),
            Bases.PowerConstant(),
        ],
    ]

    ################ Training ################

    file_name = "EnsembleDemo"
    # date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
    date_time = datetime.now().strftime("%Y-%m-%d_%H")[:-3]
    print("date_time")
    print(date_time)
    # file_path = "src/results/" + file_name + "_" + date_time + ".csv"
    file_path = "/Users/georgian/Desktop/repos/OccamNet_SocialSci/src/results/EnsembleDemo_2024-08-21.csv"

    with open(file_path, "a") as f:
        writer = csv.writer(f)

        header = [
            "wmse",
            "mse_sDev",
            "mse_median",
            "expression",
            "sDev",
            "top",
            "equalization",
            "runtime",
        ]

        writer.writerow(header)

    for sDev in sDev_sweep:
        for top in top_sweep:
            for equalization in equalization_sweep:

                print(
                    "Training with parameters: sDev={sDev}, top={top}, equalization={eq}".format(
                        sDev=sDev, top=top, eq=equalization
                    )
                )

                start = time.time()

                loss = CrossEntropyLoss(
                    sDev,
                    top,
                    anomWeight=0,
                    constantWeight=constantWeight,
                    activationWeight=activationWeight,
                )

                sparsifier = SNS()

                n = NetworkConstants(
                    inputSize,
                    layers,
                    outputSize,
                    sparsifier,
                    loss,
                    learningRate,
                    constantLearningRate,
                    temp,
                    endTemp,
                    equalization,
                    skipConnections=True,
                )

                n.setConstants([0 for j in range(n.totalConstants)])

                train_function = n.trainFunction(
                    epochs,
                    batchesPerEpoch,
                    sampleSize,
                    decay,
                    X,
                    Y,
                    useMultiprocessing=True,
                    numProcesses=20,
                    ensemble=ensembleMode,
                )

                ### Evaluation ###

                output = n.forwardFitConstants(train_function, X, Y, ensemble=True)
                output = output.squeeze(1)

                # Weighted MSE
                MSELoss = nn.MSELoss()
                losses = []

                for curr_Y in Y:
                    curr_out, output = torch.split(
                        output, [curr_Y.shape[0], output.shape[0] - curr_Y.shape[0]]
                    )
                    losses.append(MSELoss(curr_Y, curr_out).item())

                weighted_mse = np.mean(losses)
                std_mse = np.std(losses)
                median_mse = np.median(losses)

                expression = str(n.applySymbolicConstant(train_function))

                end = time.time()
                minutes = (end - start) / 60

                with open(file_path, "a") as f:
                    writer = csv.writer(f)

                    data = [
                        weighted_mse,
                        std_mse,
                        median_mse,
                        expression,
                        sDev,
                        top,
                        equalization,
                        minutes,
                    ]

                    writer.writerow(data)

    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Convert the elapsed time into hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Get the current date
    current_date = time.strftime("%Y-%m-%d")

    # Print the summary
    print(f"Date: {current_date}")
    print(f"Elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
