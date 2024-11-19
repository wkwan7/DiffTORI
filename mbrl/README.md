# DiffTORI: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning
----

Original implementation of **DiffTORI** on model-based reinforcement learning from

[DiffTORI: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning](https://arxiv.org/abs/2402.05421)

## Instructions

Assuming that you already have [MuJoCo](http://www.mujoco.org) installed, install dependencies using `conda`:

```
conda env create -f environment.yaml
conda activate difftori
```

After installing dependencies, you can train an agent by calling

```
python src/train.py task=finger-turn-easy mode='back_plan' modality=pixels device='cpu' seed=0
```
See `tasks.txt` for a list of supported tasks. The training script supports both local logging as well as cloud-based logging with [Weights & Biases](https://wandb.ai). To use W&B, provide a key by setting the environment variable `WANDB_API_KEY=<YOUR_KEY>` and add your W&B project and entity details to `cfgs/default.yaml`.


## Troubleshooting
```
RuntimeError: There was an error while running the linear optimizer. Original error message: linalg.cholesky: (Batch element 0): The factorization could not be completed because the input is not positive-definite (the leading minor of order 1 is not positive-definite).. Backward pass will not work. To obtain the best solution seen before the error, run with torch.no_grad()
```
This sometimes occurs during the training of DiffTORI. It's due to the Hessian matrix corresponding to the cost function not being positive-definite ("The factorization could not be completed because the input is not positive-definite"). A temporary solution is to increase the damping term in the Levenberg-Marquardt method or to reload the checkpoint from where the training was interrupted and continue training. In the future, using a better optimizer might avoid this problem.


## Acknowledgement
The code base used in this project is sourced from:

[nicklashansen/tdmpc](https://github.com/nicklashansen/tdmpc)

We thank the authors for their amazing implementations.


## Citation
If you use our method or code in your research, please consider citing the paper as follows:

```
@article{wan2024difftop,
  title={DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning},
  author={Wan, Weikang and Wang, Yufei and Erickson, Zackory and Held, David},
  journal={arXiv preprint arXiv:2402.05421},
  year={2024}
}
```