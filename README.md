# Electromagnetic neural source imaging under sparsity constraints with SURE-based hyperparameter tuning.

## Summary

This repository contains a full-fledged implementation of a SURE-based hyperparameter tuning technique used
to automatically finetune the regularization hyperparameter of Lasso-like models for the electromagnetic neural 
source imaging inverse problem. 

The associated paper was accepted at the workshop **Medical Imaging Meets NeurIPS 2021**.

Paper available: https://hal.archives-ouvertes.fr/hal-03418092/document

The repository also contains the code to benchmark other hyperparameter tuning methods namely, **Hierarchical Bayesian Modelling**
(Lambda-MAP) and **Spatial Cross-Validation**.

## Install

At the root of the repo, run:

```
pip install -e .
```

## Requirements

```
numpy
matplotlib
mne
```

## Miscellaneous

A working implementation of Monte Carlo Finite Difference (MCFD) SURE can be found in **MNE** (https://github.com/mne-tools/mne-python).
It is the default option when using the `mixed_norm` solver.
