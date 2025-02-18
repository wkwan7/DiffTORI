from utilis.config import Config

dmc_config = Config({
    "project_name": "BAC-v0",
    "seed": 0,
    "tag": "dmc",
    "start_steps": 5e3,
    "cuda": True,
    "num_steps": 1000001,
    "device": 0,

    "env_name": "DogRun-v0", 
    "eval": True,
    "eval_episodes": 10,
    "eval_times": 10,
    "replay_size": 1000000,

    "policy": "Gaussian",   
    "gamma": 0.99, 
    "tau": 0.005,
    "lr": 0.0003,
    "alpha": 0.2,
    "quantile": 0.9,
    "automatic_entropy_tuning": True,
    "batch_size": 512, 
    "updates_per_step": 1,
    "target_update_interval": 2,
    "hidden_size": 512,
    "msg": "default"
})

default_config = Config({
    "project_name": "BAC-v0",
    "seed": 0,
    "tag": "default",
    "start_steps": 5e3,
    "cuda": True,
    "num_steps": 1000001,
    "device": 0,

    "env_name": "HumanoidStandup-v2", 
    "eval": True,
    "eval_episodes": 10,
    "eval_times": 10,
    "replay_size": 1000000,

    "policy": "Gaussian",   # 'Policy Type: Gaussian | Deterministic (default: Gaussian)'
    "gamma": 0.99, 
    "tau": 0.005,
    "lr": 0.0003,
    "alpha": 0.2,
    "quantile": 0.9,
    "automatic_entropy_tuning": False,
    "batch_size": 256, 
    "updates_per_step": 1,
    "target_update_interval": 1,
    "hidden_size": 256,
    "msg": "default"
})

