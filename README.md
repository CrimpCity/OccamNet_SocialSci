# OccamNet for Social Science

This repository contains the codebase used for the paper [AI-Assisted Discovery of Quantitative and Formal Models in Social Science](https://arxiv.org/abs/2210.00563). It includes an OccamNet implementation and demo files which show how it is used for experiments.

The OccamNet implementation used for this work is found in the `occamnet` folder. This is a Pytorch implementation and is a slightly modified version of [this OccamNet Implementation](https://github.com/druidowm/OccamNet_Public/tree/main/implicit). OccamNet was originally described in the paper [Fast Neural Models for Symbolic Regression at Scale](https://arxiv.org/abs/2007.10784), with the corresponding GitHub repository available [here](https://github.com/druidowm/OccamNet_Public).

# Demos

The `ensemble_demo.py` file demonstrates how to use OccamNet in its most general setting to fit an ensemble of datasets.

    python ensemble_demo.py

We also include demos for our experiments with synthetic data. The `lotka_volterra_demo.py` and `sir_demo.py` files demonstrate differential equation fitting, where each equation in the ODE system is fit individually. The `solow_demo.py` file demonstrates ensemble fitting with unit checking. All demos specify hyperparameters described in the appendix of the [paper](https://arxiv.org/abs/2210.00563).

## Lotka-Volterra model

    python lotka_volterra_demo.py --target_var H
    python lotka_volterra_demo.py --target_var L

## SIR model

    python lotka_volterra_demo.py --target_var s
    python lotka_volterra_demo.py --target_var i
    python lotka_volterra_demo.py --target_var r


## Solow model

    python solow_demo.py


# All OccamNet repositories

Please note that there are many other implementations of OccamNet available. [This repository](https://github.com/druidowm/OccamNet_Versions) breifly describes the different implementations available and provides reccomendations for which one to use.

If you are a Social Scientist looking to use OccamNet in your work, this implementation is currently the best to use. It offers more functionality than other implementations at the cost of being somewhat slow. We are currently working on a new implementation which will be substantially faster. Stay tuned!

# Questions about OccamNet?

Feel free to contact Owen Dugan at odugan@mit.edu.