# Automatic hyperparameter selection for Lasso-like models solving inverse problems

## Summary

We propose a hyperparameter selection technique based on SURE to automatically calibrate
Lasso-like models solving inverse problems.

This repository contains the automatic calibrator and is used to demonstrate the
superiority of this method on sparse models solving the M/EEG inverse problem. As part
of our benchmark, we provide an implementation of two competitors: Lambda-MAP and
temporal cross-validation.

An in-depth explanation can be found here: https://arxiv.org/abs/2112.12178

This work was accepted at the **Medical Imaging Meets NeurIPS 2021** workshop.

Note that the default solver in [**MNE-Python**](https://github.com/mne-tools/mne-python)
for inverse problems is automatically calibrated using Monte Carlo Finite Difference (MCFD)
SURE.

## Installation

Start by installing the necessary requirements. We recommend creating a new `venv` or
`conda` environment. Once created and activated, run

```bash
pip install -r requirements.txt
```

Then to install our package:

```bash
pip install -e .
```
