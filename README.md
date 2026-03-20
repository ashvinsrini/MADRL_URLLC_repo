# MADRL\_URLLC\_repo



\## Repository overview



This repository contains the reference code for the core implementation of the proposed MADRL framework.



\### Main scripts

\- `LSTM\_predict.py` : trains and evaluates the interference-power prediction module.

\- `SINR\_cdf\_CI\_runner.py` : generates the SINR CDF results with confidence intervals.

\- `DRL\_async\_ci\_train.py` : runs the  DRL training/evaluation pipeline.

\- `eval\_from\_saved\_results.py` : quickly regenerates the reported figures from the provided saved result files.



\### Notes

\- The training scripts are provided as reference implementations of the core pipeline.

\- The saved result files are the results used by `eval\_from\_saved\_results.py` for plot generation.

\- Due to the stochastic and computationally intensive nature of DRL training, rerunning the training scripts may lead to minor numerical differences, while the overall behavior should remain consistent.

