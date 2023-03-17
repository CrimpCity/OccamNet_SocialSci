# OccamNet for Social Science

This repository contains the codebase used for the paper [AI-Assisted Discovery of Quantitative and Formal Models in Social Science](https://arxiv.org/abs/2210.00563). It includes an OccamNet implementation and demo files which show how it is used for experiments.

The OccamNet implementation used for this work is found in the `occamnet` folder. This is a Pytorch implementation and is a slightly modified version of [this OccamNet Implementation](https://github.com/druidowm/OccamNet_Public/tree/main/implicit). OccamNet was originally described in the paper [Fast Neural Models for Symbolic Regression at Scale](https://arxiv.org/abs/2007.10784), with the corresponding GitHub repository available [here](https://github.com/druidowm/OccamNet_Public).

# System requirements

## Hardware requirements

This implementation requires only a standard computer with enough RAM to support the in-memory operations. For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB

CPU: 4+ cores

The experiments in the paper were run on the MIT Supercloud Computing Cluster with one Intel-Xeon-Gold-6248 node with 20 CPU cores and 384 GB of RAM. 

## OS requirements

This implementation has been tested on Linux (Ubuntu 18.04.6) and macOS (Monterey 12.6) operating systems.

## Python dependencies

    matplotlib
    numpy
    pandas
    scipy
    sympy
    torch

# Demos

## Ensemble demo

The `ensemble_demo.py` file demonstrates how to use OccamNet in its most general setting to fit an ensemble of datasets. As an example, we generate 5 datasets according to the equation $y = ax^b$. OccamNet is then able to recover the correct functional form and identify the constant parameters $a$ and $b$ for each dataset in the ensemble. The average runtime for the demos is 30 minutes.

    python ensemble_demo.py

## Synthetic data experiments
We include demos for the experiments in the [paper](https://arxiv.org/abs/2210.00563) that use synthetically generated data. The `lotka_volterra_demo.py` and `sir_demo.py` files demonstrate differential equation fitting, where each equation in the ODE system is fit individually. The `solow_demo.py` file demonstrates ensemble fitting with unit checking. All demos use the hyperparameters specified in the appendix of the paper.

### Lotka-Volterra model example
$$
\begin{align*}
    \frac{dH}{dt} &= 0.03H-0.001HL\\
    \frac{dL}{dt} &= 0.006HL-0.15L
\end{align*}
$$

    python lotka_volterra_demo.py --target_var H
    python lotka_volterra_demo.py --target_var L

### SIR model example

$$
\begin{align*}
    \frac{ds}{dt} &= -0.5si\\
    \frac{di}{dt} &= 0.5si-0.2i\\
    \frac{dr}{dt} &= 0.2i
\end{align*}
$$

    python sir_demo.py --target_var s
    python sir_demo.py --target_var i
    python sir_demo.py --target_var r


### Solow model example

$$
\frac{dk}{dt} = sy-k(\delta+g+n)
$$

$\delta$ and $g$ are constant parameters that vary for each dataset in an ensemble of 20. 

    python solow_demo.py


# All OccamNet repositories

Please note that there are many other implementations of OccamNet available. [This repository](https://github.com/druidowm/OccamNet_Versions) briefly describes the different implementations available and provides reccomendations for which one to use.

If you are a Social Scientist looking to use OccamNet in your work, this implementation is currently the best to use. It offers more functionality than other implementations at the cost of being somewhat slow. We are currently working on a new implementation which will be substantially faster. Stay tuned!

# Questions about OccamNet?

Feel free to contact Owen Dugan at odugan@mit.edu.
