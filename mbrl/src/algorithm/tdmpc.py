import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
try:
    import algorithm.helper as h
except:
    import src.algorithm.helper as h
import theseus as th
import copy
import pdb
import time


class TOLD(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = h.enc(cfg)
		self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
		self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
		self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
		self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
		self._Q1_s, self._Q2_s = h.q_s(cfg), h.q_s(cfg)
		self.apply(h.orthogonal_init)
		if cfg.modality == 'pixels':
			for m in [self._reward, self._Q1, self._Q2, self._Q1_s.mlp, self._Q2_s.mlp]:
				m[-1].weight.data.fill_(0)
				m[-1].bias.data.fill_(0)
		else:
			for m in [self._reward, self._Q1, self._Q2, self._Q1_s, self._Q2_s]:
				m[-1].weight.data.fill_(0)
				m[-1].bias.data.fill_(0)

	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in [self._Q1, self._Q2]:
			h.set_requires_grad(m, enable)

	def h(self, obs):
		"""Encodes an observation into its latent representation (h)."""
		return self._encoder(obs)

	def next(self, z, a):
		"""Predicts next latent state (d) and single-step reward (R)."""
		x = torch.cat([z, a], dim=-1)
		return self._dynamics(x), self._reward(x)

	def pi(self, z, std=0):
		"""Samples an action from the learned policy (pi)."""
		mu = torch.tanh(self._pi(z))
		if std > 0:
			std = torch.ones_like(mu) * std
			return h.TruncatedNormal(mu, std).sample(clip=0.3)
		return mu

	def Q(self, z, a):
		"""Predict state-action value (Q)."""
		x = torch.cat([z, a], dim=-1)
		return self._Q1(x), self._Q2(x)
	
	def Q_s(self, s, a):
		"""Predict state-action value (Q)."""
		if self.cfg.modality == 'pixels':
			return self._Q1_s(s, a), self._Q2_s(s, a)
		else:
			x = torch.cat([s, a], dim=-1)
			return self._Q1_s(x), self._Q2_s(x)


class TDMPC():
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.std = h.linear_schedule(cfg.std_schedule, 0)
		self.model = TOLD(cfg).cuda()
		self.model_target = deepcopy(self.model)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		self.optim_a = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
		self.aug = h.RandomShiftsAug(cfg)
		self.model.eval()
		self.model_target.eval()
		self.total_a_loss = 0

	def state_dict(self):
		"""Retrieve state dict of TOLD model, including slow-moving target network."""
		return {'model': self.model.state_dict(),
				'model_target': self.model_target.state_dict()}

	def save(self, fp):
		"""Save state dict of TOLD model to filepath."""
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		"""Load a saved state dict from filepath into current agent."""
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])

	@torch.no_grad()
	def estimate_value(self, z, actions, horizon):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			z, reward = self.model.next(z, actions[t])
			G += discount * reward
			discount *= self.cfg.discount
		G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
		return G

	@torch.no_grad()
	def plan(self, obs, eval_mode=False, step=None, t0=True):
		"""
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		# Seed steps
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

		# Sample policy trajectories
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
			z = self.model.h(obs).repeat(num_pi_trajs, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z, self.cfg.min_std)
				z, _ = self.model.next(z, pi_actions[t])

		# Initialize state and parameters
		z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
		mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
		std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
		if not t0 and hasattr(self, '_prev_mean'):
			mean[:-1] = self._prev_mean[1:]

		# Iterate CEM
		for i in range(self.cfg.iterations):
			actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
			if num_pi_trajs > 0:
				actions = torch.cat([actions, pi_actions], dim=1)

			# Compute elite actions
			value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			_std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
			_std = _std.clamp_(self.std, 2)
			mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

		# Outputs
		score = score.squeeze(1).cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		self._prev_mean = mean
		mean, std = actions[0], _std[0]
		a = mean
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
		return a

	def update_pi(self, zs):
		"""Update policy using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		# Loss is a weighted sum of Q-values
		pi_loss = 0
		for t,z in enumerate(zs):
			a = self.model.pi(z, self.cfg.min_std)
			Q = torch.min(*self.model.Q(z, a))
			pi_loss += -Q.mean() * (self.cfg.rho ** t)

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_obs, reward):
		"""Compute the TD-target from a reward and the observation at the following time step."""
		next_z = self.model.h(next_obs)
		td_target = reward + self.cfg.discount * \
			torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
		return td_target
	
	@torch.no_grad()
	def _td_target_s(self, next_obs, reward):
		"""Compute the TD-target from a reward and the observation at the following time step."""
		next_z = self.model.h(next_obs)
		td_target = reward + self.cfg.discount * \
			torch.min(*self.model_target.Q_s(next_obs, self.model.pi(next_z, self.cfg.min_std)))
		return td_target

	def update(self, replay_buffer, step):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		for params in self.model.parameters():
			if params.grad is not None:
				params.grad.zero_()
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self.model.train()
		
		# Representation
		z = self.model.h(self.aug(obs))
		zs = [z.detach()]

		consistency_loss, reward_loss, value_loss, value_s_loss, priority_loss = 0, 0, 0, 0, 0

		# value_s_loss
		aug_obs = self.aug(obs)
		for t in range(self.cfg.horizon):
			Q1, Q2 = self.model.Q_s(aug_obs, action[t])
			with torch.no_grad():
				next_obs = self.aug(next_obses[t])
				td_target = self._td_target_s(next_obs, reward[t])
			aug_obs = next_obs

			rho = (self.cfg.rho ** t)
			value_s_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))

		for t in range(self.cfg.horizon):
			# Predictions
			Q1, Q2 = self.model.Q(z, action[t])
			z, reward_pred = self.model.next(z, action[t])
			with torch.no_grad():
				next_obs = self.aug(next_obses[t])
				next_z = self.model_target.h(next_obs)
				td_target = self._td_target(next_obs, reward[t])
			zs.append(z.detach())

			# Losses
			rho = (self.cfg.rho ** t)
			consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
			reward_loss += rho * h.mse(reward_pred, reward[t])
			value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
			priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))
		
		# Optimize model
		total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_s_loss.clamp(max=1e4)
		weighted_loss = (total_loss.squeeze(1) * weights).mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

		# Update policy + target network
		pi_loss = self.update_pi(zs)
		if step % self.cfg.update_freq == 0:
			h.ema(self.model, self.model_target, self.cfg.tau)
		
		self.model.eval()
		loss_info = {'consistency_loss': float(consistency_loss.mean().item()),
				'reward_loss': float(reward_loss.mean().item()),
				'value_loss': float(value_loss.mean().item()),
				'value_s_loss': float(value_s_loss.mean().item()),
				'pi_loss': pi_loss,
				'total_loss': float(total_loss.mean().item()),
				'weighted_loss': float(weighted_loss.mean().item()),
				'grad_norm': float(grad_norm)}
		return loss_info
	
	def plan_theseus(self, obs, eval_mode=False, step=None, t0=True):
		"""
		Plan next action using trajectory optimization in theseus without backpropagation.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		for params in self.model.parameters():
			if params.grad is not None:
				params.grad.zero_()
		# Seed steps
		if step < self.cfg.seed_steps: #and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

		# Initialize state and parameters
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		z = self.model.h(obs).repeat(1, 1) # batch size(1), laten state dim
		actions = torch.zeros(horizon, self.cfg.action_dim, device=self.device) # batch size(1), horizon*action dim
		if not t0 and hasattr(self, '_prev_actions'):
			actions[:-1] = self._prev_actions[1:]
		actions = actions.view(1,-1)

		with torch.no_grad():
			pi_actions = torch.empty(horizon, 1, self.cfg.action_dim, device=self.device)
			z0 = self.model.h(obs).repeat(1, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z0, self.cfg.min_std)
				z0, _ = self.model.next(z0, pi_actions[t])
		pi_actions = pi_actions.view(1,-1)
		#init_actions = copy.deepcopy(actions)
		init_actions = copy.deepcopy(pi_actions)
		
		# theseus solver
		actions = th.Vector(tensor=actions, name="actions")
		z = th.Variable(z, name="z")

		def value_cost_fn(optim_vars, aux_vars):
			actions = optim_vars[0].tensor # batch size(1), horizon*action dim
			actions = actions.view(horizon,1,self.cfg.action_dim)
			z = aux_vars[0].tensor
			tmp = torch.clamp(torch.abs(actions)-1, min=0, max=1e4)
			actions = torch.clamp(actions, -1, 1,)  # action clipping
			G, discount = 0, 1
			for t in range(horizon):
				z, reward = self.model.next(z, actions[t])
				G += discount * reward
				discount *= self.cfg.discount
			G += discount * torch.min(*self.model.Q(z, self.model.pi(z, 0)))
			err = -G.nan_to_num_(0) + 1000
			return err

		optim_vars = [actions]
		aux_vars = [z]
		cost_function = th.AutoDiffCostFunction(
			optim_vars, value_cost_fn, 1, aux_vars=aux_vars, name="value_cost_fn", 
		)
		objective = th.Objective()
		objective.to(device=self.device, dtype=torch.float32)
		objective.add(cost_function)
		optimizer = th.LevenbergMarquardt(
			objective,
			th.CholeskyDenseSolver,
			max_iterations = self.cfg.max_iterations,
			step_size = self.cfg.step_size,
		)
		theseus_optim = th.TheseusLayer(optimizer)
		theseus_optim.to(device=self.device, dtype=torch.float32)
		theseus_inputs = {
		"actions": init_actions,
		"z": self.model.h(obs).repeat(1, 1),
		}
		if eval_mode:
			updated_inputs, info = theseus_optim.forward(
				theseus_inputs, optimizer_kwargs={"track_best_solution": True, 
			"verbose": False, "damping": self.cfg.damping, "backward_mode": self.cfg.backward_mode, "backward_num_iterations": self.cfg.backward_num_iterations,})
			# print("Best solution:", info.best_solution)
			# print("Optim error: ", info.last_err.item())
		else:
			updated_inputs, info = theseus_optim.forward(
				theseus_inputs, optimizer_kwargs={"track_best_solution": True, 
			"verbose": False, "damping": self.cfg.damping, "backward_mode": self.cfg.backward_mode, "backward_num_iterations": self.cfg.backward_num_iterations,})
		for params in self.model.parameters():
			if params.grad is not None:
				params.grad.zero_()
			
		best_actions = info.best_solution['actions']

		best_actions = best_actions.view(horizon,-1)
		self._prev_actions = best_actions
		a = best_actions[0].nan_to_num_(0)

		return a

	def plan_theseus_update(self, obs, eval_mode=False, step=None, t0=True):
		"""
		Plan next action using trajectory optimization in theseus and backpropagate.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		loss_info = {}
		# Seed steps
		if step < self.cfg.seed_steps: #and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1), info
		
		if t0:
			self.optim_a.zero_grad(set_to_none=True)
			for params in self.model.parameters():
				if params.grad is not None:
					params.grad.zero_()

		# Initialize state and parameters
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) # batch size(1), obs dim
		obs_copy = copy.deepcopy(obs)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		z = self.model.h(obs).repeat(1, 1) # batch size(1), laten state dim
		zs = z.detach()
		actions = torch.zeros(horizon, self.cfg.action_dim, device=self.device) # batch size(1), horizon*action dim
		if not t0 and hasattr(self, '_prev_actions'):
			actions[:-1] = self._prev_actions[1:]
		actions = actions.view(1,-1)
		
		with torch.no_grad():
			pi_actions = torch.empty(horizon, 1, self.cfg.action_dim, device=self.device)
			z0 = self.model.h(obs).repeat(1, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z0, self.cfg.min_std)
				z0, _ = self.model.next(z0, pi_actions[t])
		pi_actions = pi_actions.view(1,-1)
		#init_actions = copy.deepcopy(actions)
		init_actions = copy.deepcopy(pi_actions)
		
		# theseus solver
		actions = th.Vector(tensor=actions, name="actions")
		obs = th.Variable(obs, name="obs")

		def value_cost_fn(optim_vars, aux_vars):
			actions = optim_vars[0].tensor # batch size(1), horizon*action dim
			actions = actions.view(horizon,1,self.cfg.action_dim)
			obs = aux_vars[0].tensor
			z = self.model.h(obs).repeat(1, 1)
			actions = torch.clamp(actions, -1, 1,)  # action clipping
			G, discount = 0, 1
			for t in range(horizon):
				z, reward = self.model.next(z, actions[t])
				G += discount * reward
				discount *= self.cfg.discount
			G += discount * torch.min(*self.model.Q(z, self.model.pi(z, 0)))
			err = -G.nan_to_num_(0) + 1000
			return err

		optim_vars = [actions]
		aux_vars = [obs]
		cost_function = th.AutoDiffCostFunction(
			optim_vars, value_cost_fn, 1, aux_vars=aux_vars, name="value_cost_fn", 
			#autograd_mode=th.AutogradMode.LOOP_BATCH
		)
		objective = th.Objective()
		objective.to(device=self.device, dtype=torch.float32)
		objective.add(cost_function)
		optimizer = th.LevenbergMarquardt(
			objective,
			th.CholeskyDenseSolver,
			max_iterations = self.cfg.max_iterations,
			step_size = self.cfg.step_size,
		)
		theseus_optim = th.TheseusLayer(optimizer)
		theseus_optim.to(device=self.device, dtype=torch.float32)
		theseus_inputs = {
		"actions": init_actions,
		"obs": obs,
		}
		if eval_mode:
			updated_inputs, info = theseus_optim.forward(
				theseus_inputs, optimizer_kwargs={"track_best_solution": True, 
				"verbose": False, "damping": self.cfg.damping, "backward_mode": self.cfg.backward_mode, "backward_num_iterations": self.cfg.backward_num_iterations,})
			updated_actions = updated_inputs['actions']
			best_actions = info.best_solution['actions']
		else:
			updated_inputs, info = theseus_optim.forward(
				theseus_inputs, optimizer_kwargs={"track_best_solution": True, 
				"verbose": False, "damping": self.cfg.damping, "backward_mode": self.cfg.backward_mode, "backward_num_iterations": self.cfg.backward_num_iterations,})
			updated_actions = updated_inputs['actions']
			best_actions = info.best_solution['actions']
	
		best_actions = best_actions.view(horizon,-1)
		updated_actions = updated_actions.view(horizon,-1)
		self._prev_actions = best_actions
		a = best_actions[0].nan_to_num_(0)
		a_t = updated_actions[0].nan_to_num_(0)

		if eval_mode:
			return a, loss_info

		# update model
		a_t = a_t.view(1,-1)
		for params in self.model._Q1_s.parameters():
			params.requires_grad = False
		for params in self.model._Q2_s.parameters():
			params.requires_grad = False
		for params in self.model._pi.parameters():
			params.requires_grad = False
		self.a_loss = -torch.min(*self.model.Q_s(obs_copy, a_t))*1
		self.a_loss.backward()
		self.total_a_loss = self.total_a_loss + self.a_loss.clamp(max=1e4)
		
		loss_info = {'a_loss': float(self.a_loss.mean().item())}
		return a, loss_info
	
	def update2(self):
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm_a, error_if_nonfinite=False)

		self.optim_a.step()
		loss_info = {'total_a_loss': float(self.total_a_loss.mean().item()),
				'grad_norm_a': float(grad_norm)}
		for params in self.model._Q1_s.parameters():
			params.requires_grad = True
		for params in self.model._Q2_s.parameters():
			params.requires_grad = True
		for params in self.model._pi.parameters():
			params.requires_grad = True
		self.total_a_loss = 0
		self.optim_a.zero_grad(set_to_none=True)
		return loss_info