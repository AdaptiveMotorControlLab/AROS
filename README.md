# AROS: Adversarially Robust Out-of-Distribution Detection through Stability


[![PyPI version](https://badge.fury.io/py/aros-node.svg)](https://badge.fury.io/py/aros-node)
[![PyPI Downloads](https://static.pepy.tech/badge/aros-node)](https://pepy.tech/projects/aros-node)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-purple)
[![Paper](https://badgen.net/badge/icon/arXiv?icon=awesome&label&color=orange&style=flat-square)](https://arxiv.org/abs/2410.10744)

## Overview

This repository contains the code for the paper **"Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings"**. The method, termed **AROS**, employs Neural Ordinary Differential Equations (NODEs) with Lyapunov stability to create robust embeddings for OOD detection, significantly improving performance against adversarial attacks.
This repository contains the code for the paper **"Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings"**. The method, termed **AROS**, employs Neural Ordinary Differential Equations (NODEs) with Lyapunov stability to create robust embeddings for OOD detection, significantly improving performance against adversarial attacks. Additionally, the repository includes two notebooks: one demonstrates the training and evaluation process on the CIFAR-10 and CIFAR-100 datasets, while the other focuses on the ablation study.

![AROS](https://github.com/user-attachments/assets/b0d9e7f8-e39d-4bae-aee2-79a247b5e87f)

## Publication 

AROS 💍 is accepted to the **International Conference on Learning Representations 2025** (ICLR 2025)!

```
@article{mirzaei2025aros,
      title={Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings}, 
      author={Hossein Mirzaei and Mackenzie W. Mathis},
      year={2025},
      journal={The Thirteenth International Conference on Learning Representations},
      url={https://openreview.net/forum?id=GrDne4055L}, 
}
```

## Preprint

Check out our preprint now out on arXiv! [Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings](https://arxiv.org/abs/2410.10744). Hossein Mirzaeri & Mackenzie W. Mathis. Oct 2024.

 
## Key Features

- **Lyapunov Stability for OOD Detection**: Ensures that perturbed inputs converge back to stable equilibrium points, improving robustness against adversarial attacks.
- **Fake Embedding Crafting Strategy**: Generates fake OOD embeddings by sampling from the low-likelihood regions of the ID data feature space, eliminating the need for additional OOD datasets.
- **Orthogonal Binary Layer**: Enhances separation between ID and OOD embeddings, further improving robustness.

## Demo 

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdaptiveMotorControlLab/AROS/blob/main/Notebooks/AROS.ipynb) This notebook is designed to replicate and analyze the results presented in Table 1 of the AROS paper, focusing on out-of-distribution detection performance under both attack scenarios and clean evaluation.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdaptiveMotorControlLab/AROS/blob/main/Notebooks/Ablation_Study.ipynb) This notebook is designed to demo the ablation study.


## Repository Structure


- **`AROS/`**
  - **`data_loader.py`**: Contains the data loading utilities for training and evaluation.
  - **`evaluate.py`**: Implements the evaluation metrics and testing routines for the AROS model.
  - **`Main.py`**: The main script for training and testing AROS, combining all components.
  - **`stability_loss_function.py`**: Defines the Lyapunov-based loss function used for stabilizing the NODE dynamics.
  - **`utils.py`**: Includes various helper functions used throughout the project.
- **`requirements.txt`**: Lists the dependencies required to run the project.
- **`Notebooks/`**
  - **`AROS.ipynb`**
  - **`Notebooks/Ablation_Study.ipynb`**

## Installation

To install the necessary packages, run:
```python
pip install git+https://github.com/RobustBench/robustbench.git
pip install aros-node
```

To install the necessary packages from source (locally), run:

```bash
pip install -r requirements.txt
```

## Citations

ICLR:

```
@article{mirzaei2025aros,
      title={Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings}, 
      author={Hossein Mirzaei and Mackenzie W. Mathis},
      year={2025},
      journal={The Thirteenth International Conference on Learning Representations},
      url={https://openreview.net/forum?id=GrDne4055L}, 
}
```
arXiv:

```
@article{mirzaei2024aros,
      title={Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings}, 
      author={Hossein Mirzaei and Mackenzie W. Mathis},
      year={2024},
      eprint={2410.10744},
      journal={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.10744}, 
}
```


