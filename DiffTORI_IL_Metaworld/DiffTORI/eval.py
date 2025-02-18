import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import os 
from copy import deepcopy
from diffusion_policy_3d.workspace.train_diffusion_unet_hybrid_pointcloud_workspace import TrainDiffusionUnetHybridPointcloudWorkspace


os.environ['WANDB_SILENT'] = "True"

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d','config'))
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

    workspace.eval()

if __name__ == "__main__":
    main()
