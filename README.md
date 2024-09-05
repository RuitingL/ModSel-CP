# ModSel-CP

This repo provides the code for reproducing the numerical results in the paper [Conformal prediction after efficiency-oriented model selection](https://arxiv.org/abs/2408.07066).

## Simulation

- For the experiment with the residual score, the implementation is executed using `simulations_residual.py` along with helper functions defined in `mtds_func_residual.py`. 
- For the experiment with the rescaled residual score, the implementation is executed using `simulations_RescaledResidual.py` along with helper functions defined in `mtds_func_rescale_residual.py`.
- For the classification experiment, the implementation is executed using `simulations_classification.py` along with helper functions defined in `mtds_func_classification.py`.

## Real data example

We used the protein structure dataset `CASP.csv` from UCI repository https://archive.ics.uci.edu/dataset/154/protein+data.

To reproduce the result, execute the code found in `real_data_CQR.py`.

## Plots

Plots in the paper are drawn using `plot_code.py`.
