# MADRL\_URLLC\_repo



\## Repository overview



This repository contains the reference code for the core implementation of the proposed MADRL framework.



\### Main scripts

\- `LSTM_predict.py` : trains and evaluates the interference-power prediction module.

\- `SINR_cdf_CI_runner.py` : generates the SINR CDF results with confidence intervals.

\- `DRL_async_ci_train.py` : runs the  DRL training/evaluation pipeline.

\- `eval_from_saved_results.py` : quickly regenerates the reported figures from the provided saved result files.



\### Notes

\- The training scripts are provided as reference implementations of the core pipeline.

\- The saved result files are the results used by `eval\_from\_saved\_results.py` for plot generation.

\- Due to the stochastic and computationally intensive nature of DRL training, rerunning the training scripts may lead to minor numerical differences, while the overall behavior should remain consistent.

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{11174624,
  author    = {Srinivasan, Ashvin and Zhang, Junshan and Tirkkonen, Olav},
  booktitle = {2025 IEEE 101st Vehicular Technology Conference (VTC2025-Spring)},
  title     = {Asynchronous Multi-Agent Reinforcement Learning for Scheduling in Subnetworks},
  year      = {2025},
  pages     = {1--6},
  doi       = {10.1109/VTC2025-Spring65109.2025.11174624}
}


