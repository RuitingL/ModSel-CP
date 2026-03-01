# Reproducing experiments

This repo provides the code for reproducing the numerical results in the paper [Conformal prediction after data-dependent model selection](https://arxiv.org/abs/2408.07066).

## Paper description

In this paper, we address the challenge of constructing a valid prediction set after data-dependent model selection, e.g., selecting the model that minimizes the width of the resulting prediction sets. We propose two novel methods, ModSel-CP and ModSel-CP-LOO, which can be implemented efficiently and admit finite-sample validity guarantees without invoking additional sample-splitting. The efficiency of the prediction sets constructed by our methods are shown both theoretically and empirically.


## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```


## Directory Overview

```
requirements.txt                           # Dependencies
simulations_residual.py                    # Simulation with the residual score
simulations_RescaledResidual.py            # Simulation with the rescaled residual score
simulation_classification.py               # Simulation of classification
Residual experiment [2models example].py   # Simulation in the Appendix D + plotting
real_data_CQR.py                           # CQR experiment using protein dataset
realData                                   # Folder containing protein dataset
mtds_func_residual.py                      # Helper functions for experiments with the residual score
mtds_func_rescale_residual.py              # Helper functions for experiments with the rescaled residual score
mtds_func_cqrBreak.py                      # Helper functions for experiments with CQR
mtds_func_classification.py                # Helper functions for the classification experiment
plots for regression with standard error bar.py      # Plotting for regression
plots for classification with standard error bar.py  # Plotting for classification
```


## Simulation

- For the experiment with the **residual score**, run `simulations_residual.py`. 
- For the experiment with the **rescaled residual score**, run `simulations_RescaledResidual.py`.
- For the **classification** experiment, run `simulations_classification.py`.
- For the additional simulation in the Appendix D, run `Residual experiment [2models example].py`.


## Real data example

- **Dataset**: Protein structure dataset `CASP.csv` from UCI repository https://archive.ics.uci.edu/dataset/154/protein+data.
- **Script**: `real_data_CQR.py`.


## Code for plotting

- **Regression** results with standard error bar: `plots for regression with standard error bar.py`

- **Classification** results with standard error bar: `plots for classification with standard error bar.py`
