# DiffTORI: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning
<a href="https://arxiv.org/abs/2402.05421"><strong>Paper</strong></a>
|
<a href="https://x.com/weikang_wan/status/1866644189523611720"><strong>Twitter</strong></a> 

<a href="https://wkwan7.github.io/">Weikang Wan*</a>, 
<a href="https://wadiuvatzy.github.io/">Ziyu Wang*</a>, 
<a href="https://yufeiwang63.github.io/">Yufei Wang*</a>, 
<a href="https://zackory.com/">Zackory Erickson</a>, 
<a href="https://davheld.github.io/">David Held</a>

This repository contains the official implementation of **DiffTORI**, which utilizes Differentiable Trajectory Optimization as the policy representation to generate actions for deep Reinforcement and Imitation learning.

## Repository Overview

This project is organized into two main components:

- **mbrl**  
  This folder contains the code for model-based reinforcement learning using DiffTORI. For detailed instructions on installation, training, and troubleshooting, please refer to the `mbrl/README.md`.

- **DiffTORI_IL_Metaworld**  
  This folder provides the implementation of DiffTORI for imitation learning for Metaworld tasks. Detailed guidance on data generation, training, and evaluation can be found in the `DiffTORI_IL_Metaworld/README.md`.

## Citation

For further details on the methodology, please refer to our paper:  
[DiffTORI: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning](https://arxiv.org/abs/2402.05421)

If you find our work useful, please consider citing:

```bibtex
@article{wan2024difftop,
  title={DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning},
  author={Wan, Weikang and Wang, Yufei and Erickson, Zackory and Held, David},
  journal={arXiv preprint arXiv:2402.05421},
  year={2024}
}
