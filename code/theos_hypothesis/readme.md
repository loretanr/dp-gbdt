# Theos hypothesis

This directory contains the code to generate measurements and graphs related to "Theos hypothesis / second-split" DP-GBDT.

The basis is [https://gitlab.inf.ethz.ch/kkari/ppml-insurance](https://gitlab.inf.ethz.ch/kkari/ppml-insurance), but only the relevant files are taken over and all unrelated content of the repository was removed.

The code is to be run from this directory, e.g. `python results/abalone/cross_val_2ndsplit.py`. And the model you want to test should be named `model.py`, so do something like `cp DPGBDT/model_2ndsplit.py DPGBDT/model.py` first.
