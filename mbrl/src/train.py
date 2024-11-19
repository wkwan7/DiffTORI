import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ['CUDA_VISIBLE_DEVICES']='3'
#os.environ['EGL_DEVICE_ID']='6'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'
import pdb

# MODE = ""
# "default"(TD-MPC); "theseus_forward", "back_plan"(after whole episdoe, theseus);  "back_plan_per_step"(theseus)s

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def evaluate(env, agent, num_episodes, step, env_step, video, mode):
	"""Evaluate a trained agent and optionally save a video."""
	MODE = mode
	episode_rewards, episode_success = [], []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if video: video.init(env, enabled=(i==0))
		while not done:
			if MODE == "default":
				action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			elif MODE == "back_plan":
				if step>=6000:
					action, info = agent.plan_theseus_update(obs, eval_mode=True, step=step, t0=t==0)
				else:
					action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			elif MODE == "theseus_forward":
				action = agent.plan_theseus(obs, eval_mode=True, step=step, t0=t==0)
			obs, reward, done, _info = env.step(action.cpu().numpy())
			ep_reward += reward
			if video: video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		episode_success.append(int(_info.get('success', 0)))
		if video: video.save(env_step)
	return np.nanmean(episode_rewards), np.nanmean(episode_success)


def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	MODE = cfg.mode
	cfg.train_steps = int(cfg.train_steps / 3) #pixel-based only run 1000k step
	work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
	
	# Run training
	L = logger.Logger(work_dir, cfg)
	episode_idx, start_time = 0, time.time()
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
		train_metrics = {}
		# Collect trajectory
		obs = env.reset()
		episode = Episode(cfg, obs)
		episode_step = 1
		while not episode.done:
			if MODE == "default":
				action = agent.plan(obs, step=step, t0=episode.first)
			elif MODE == "back_plan":
				if step>=6000:
					action, info = agent.plan_theseus_update(obs, step=step, t0=episode.first)
					if info:
						train_metrics.update(info)
				else:
					action = agent.plan(obs, step=step, t0=episode.first)
			elif MODE == "theseus_forward":
				action = agent.plan_theseus(obs, step=step, t0=episode.first)

			
			obs, reward, done, _ = env.step(action.cpu().numpy())
			episode += (obs, action, reward, done)
			episode_step += 1
			if MODE == "back_plan" and episode.done and step>=6000:
				train_metrics.update(agent.update2())

		assert len(episode) == cfg.episode_length
		buffer += episode

		# Update model
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))

		# Log training episode
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			common_metrics['episode_reward'], common_metrics['episode_success'] = \
				evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video, MODE)
			L.log(common_metrics, category='eval')

	L.finish(agent)
	print('Training completed successfully')


if __name__ == '__main__':
	train(parse_cfg(Path().cwd() / __CONFIG__))
