# AROS: Adversarially Robust Out-of-Distribution Detection through Stability

## Overview

This repository contains the code for the paper **"Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings"**. The method, termed **AROS**, employs Neural Ordinary Differential Equations (NODEs) with Lyapunov stability to create robust embeddings for OOD detection, significantly improving performance against adversarial attacks. An example of training and evaluation of the model on the  CIFAR-10 and CIFAR-100 of both benchmark  is available in this [notebook]([url](https://colab.research.google.com/drive/1-VrfWbnlW_2x_lybVfyCD70OOEelrSYB?usp=sharing)).
This repository contains the code for the paper **"Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings"**. The method, termed **AROS**, employs Neural Ordinary Differential Equations (NODEs) with Lyapunov stability to create robust embeddings for OOD detection, significantly improving performance against adversarial attacks. An example of training and evaluation of the model on the  CIFAR-10 and CIFAR-100 of both benchmark  is available in this [notebook]().



![AROS](https://github.com/user-attachments/assets/dd5d5dd9-2650-4746-9983-5abf6d7eedfc)

## Key Features

- **Lyapunov Stability for OOD Detection**: Ensures that perturbed inputs converge back to stable equilibrium points, improving robustness against adversarial attacks.
- **Fake Embedding Crafting Strategy**: Generates fake OOD embeddings by sampling from the low-likelihood regions of the ID data feature space, eliminating the need for additional OOD datasets.
- **Orthogonal Binary Layer**: Enhances separation between ID and OOD embeddings, further improving robustness.

## Repository Structure


- **AROS/**
  - **`data_loader.py`**: Contains the data loading utilities for training and evaluation.
  - **`evaluate.py`**: Implements the evaluation metrics and testing routines for the AROS model.
  - **`Main.py`**: The main script for training and testing AROS, combining all components.
  - **`stability_loss_function.py`**: Defines the Lyapunov-based loss function used for stabilizing the NODE dynamics.
  - **`utils.py`**: Includes various helper functions used throughout the project.
- **`requirements.txt`**: Lists the dependencies required to run the project.



## Installation

To install the necessary packages, run:

```bash
pip install -r requirements.txt
