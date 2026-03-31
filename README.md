# Regression on Parkinson's Disease Data

Python project for predicting **total UPDRS** from Parkinson's telemonitoring voice measurements using multiple regression approaches.

## Overview

This repository contains an academic machine learning project focused on regression on Parkinson's disease data. The workflow:

- loads and preprocesses patient measurements from `parkinsons_updrs.csv`
- aggregates repeated measurements by `subject#` and integer `test_time`
- normalizes the dataset using statistics computed on the training split only
- compares three regression strategies:
  - **Linear Least Squares (LLS)**
  - **Steepest Descent (SD)**
  - **Local Linear Regression (LLR)** based on nearest neighbors
- evaluates the models with error statistics, correlation, and `R^2`
- generates diagnostic plots for features, weights, errors, and predictions

The project also includes a PDF report documenting the work.

## Methods Implemented

### 1. Linear Least Squares
A closed-form linear regression baseline is trained on the normalized training set.

### 2. Steepest Descent
A gradient-based optimizer is used to estimate the regression weights iteratively.

### 3. Local Linear Regression
For each sample, the method selects the `k` nearest training points and fits a local model. The script also studies how the test MSE changes as `k` varies.

## Preprocessing

The current code performs the following preprocessing steps:

- converts `test_time` to integer values
- groups samples by `subject#` and `test_time`, averaging duplicated entries
- shuffles the dataset using a seed
- uses a **50/50 train-test split**
- standardizes features using **training-set mean and standard deviation**
- predicts `total_UPDRS`
- removes `subject#`, `Jitter:DDP`, and `Shimmer:DDA` from the regressors

## Repository Structure

```text
Regression-on-Parkinson-s-disease/
├── Regression on Parkinson’s disease data.pdf
└── Regression on Parkinson’s disease data_code/
    ├── main.py
    └── sub/
        ├── Minimization.py
        ├── Seeds.py
        ├── myCustomClasses.py
        └── myCustomPlots.py
```

## Requirements

Install the required packages with:

```bash
pip install numpy pandas matplotlib
```

## Dataset

The script expects a file named:

```text
parkinsons_updrs.csv
```

Place it in the same directory where `main.py` is executed.

## How to Run

From the code folder:

```bash
cd "Regression on Parkinson’s disease data_code"
python main.py
```

At the moment, the main script uses a hardcoded seed-like identifier:

```python
matricola = 247586
```

You can change it directly inside `main.py`, or modify the script later to accept a command-line argument.

## Output Files

Depending on the executed path, the project generates plots and result files such as:

- `corr_coeff.png`
- `UPDRS_corr_coeff.png`
- `LLS-what.png`
- `SD-what.png`
- `LLS-hist.png`
- `SD-hist.png`
- `LOCAL-hist.png`
- `LLS-yhat_vs_y.png`
- `SD-yhat_vs_y.png`
- `LOCAL-yhat_vs_y.png`
- `K-MSE.png`
- `results_LLR.csv`

The script also prints regression metrics for training and test sets, including:

- min / max error
- mean and standard deviation of the error
- MSE
- `R^2`
- correlation coefficient

## Notes

This repository is a good academic project for showing:

- regression fundamentals
- data preprocessing for biomedical measurements
- comparison between analytical and iterative optimization
- local modeling with nearest neighbors
- result visualization with Matplotlib

## Suggested Next Improvements

- add the missing dataset or document its source explicitly
- add a `requirements.txt`
- rename folders to remove spaces and special characters
- replace the hardcoded `matricola` with a command-line argument
- turn `sub/` into a proper Python package
- save all model results, not only `results_LLR.csv`
- add example output images to this README
