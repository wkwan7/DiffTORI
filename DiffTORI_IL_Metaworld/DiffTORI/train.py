"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import os
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import pathlib
from omegaconf import OmegaConf
import hydra
from termcolor import cprint
from copy import deepcopy
from diffusion_policy_3d.workspace.train_diffusion_unet_hybrid_pointcloud_workspace import TrainDiffusionUnetHybridPointcloudWorkspace
import time

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['WANDB_SILENT'] = "True"

# allow for detecting segmentation fault
# import faulthandler
# faulthandler.enable()
# cprint("[fault handler enabled]", "cyan")

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    if cfg.policy.expert_policy is not None:
        temp_cfg = deepcopy(cfg)
        temp_cfg.policy._target_ = 'diffusion_policy_3d.policy.diffusion_unet_hybrid_pointcloud_policy.DiffusionUnetHybridPointcloudPolicy'
        temp_cfg.training.use_ema = True
        workspace = TrainDiffusionUnetHybridPointcloudWorkspace(temp_cfg)
        workspace.load_checkpoint(path=cfg.policy.expert_policy)
        policy = deepcopy(workspace.model)
        policy.eval()
    # import pdb; pdb.set_trace()

    workspace = TrainDiffusionUnetHybridPointcloudWorkspace(cfg)
    if cfg.policy.expert_policy is not None:
        workspace.model.set_expert_policy(policy)
        if cfg.training.use_ema:
            workspace.ema_model.expert_policy = deepcopy(workspace.model.expert_policy)

    # cls = hydra.utils.get_class(cfg._target_)
    # workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    # cprint("sleeping for 2 hours", "cyan")
    # time.sleep(3600 * 2)
    main()
