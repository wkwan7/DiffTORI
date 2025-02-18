import torch
import torch.nn as nn
from copy import deepcopy
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.vision_3d.pointnet_extractor import DP3Encoder
import theseus as th

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer_dims, activation=nn.ELU):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dims = layer_dims
        self.activation = activation
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for dim in layer_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
            self.layers.append(activation())
        self.layers.append(nn.Linear(prev_dim, output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DiffTORI(nn.Module):
    def __init__(self, 
                shape_meta: dict,
                horizon, 
                n_action_steps, 
                n_obs_steps,
                encoder_output_dim=256,
                crop_shape=None,
                use_pc_color=False,
                pointcloud_encoder_cfg=None,
                soft_update_tau=0.01,
                **kwargs
                ):
        super(DiffTORI, self).__init__()
        self.cfg = None
        self.planning_horizon = 1

        self.traj_opt_step = kwargs['traj_opt_step']
        self.damping = kwargs['damping']
        self.expert_noise = kwargs['expert_noise']
        self.traj_opt_num = int(kwargs['traj_opt_num'])
        self.action_loss_weight = kwargs['action_loss_weight']
        self.mlp_hidden_dim = int(kwargs['mlp_hidden_dim'])
        self.use_zero_initial = kwargs['use_zero_initial']
        print("=============== expert_noise =============== ", self.expert_noise)
        print("=============== traj_opt_step =============== ", self.traj_opt_step)
        print("=============== damping =============== ", self.damping)
        print("=============== traj_opt_num =============== ", self.traj_opt_num)
        print("=============== action_loss_weight =============== ", self.action_loss_weight)
        print("=============== mlp_hidden_dim =============== ", self.mlp_hidden_dim)

        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            self.action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            self.action_dim = action_shape[0] * action_shape[1]
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        self.obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type='pointnet',
                                                )
        self.obs_feature_dim = self.obs_encoder.output_shape()

        

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.use_pc_color = use_pc_color
        self.z_dim = 64
        self._create_cvae_networks()

        self.optimizer = torch.optim.Adam([
            {'params': self.obs_encoder.parameters()},
            {'params': self.cvae_encoder.parameters()},
            {'params': self.reward_network.parameters()},
        ], lr=1e-4)
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=15000,
            eta_min=1e-6
        )
        

        self.nets = nn.ModuleDict()
        self.nets['model'] = nn.ModuleDict()
        self.nets['model']['pc_encoder'] = self.obs_encoder
        self.nets['model']['cvae_encoder'] = self.cvae_encoder
        self.nets['model']['reward'] = self.reward_network

        self.nets = self.nets.float().to(self.device)

        self.soft_update_tau = soft_update_tau
        self.expert_policy = None

        self.train_batch_size = 128

    def set_expert_policy(self, expert_policy):
        self.expert_policy = deepcopy(expert_policy)
        self.expert_policy.eval()

    def _create_cvae_networks(self):
        self.cvae_encoder_input_dim = self.obs_feature_dim * self.n_obs_steps + self.action_dim * self.horizon
        self.cvae_encoder_output_dim = self.z_dim * 2
        self.cvae_encoder_layer_dims = [self.mlp_hidden_dim, self.mlp_hidden_dim]
        self.cvae_encoder = MLP(
            input_dim=self.cvae_encoder_input_dim,
            output_dim=self.cvae_encoder_output_dim,
            layer_dims=self.cvae_encoder_layer_dims,
        )

        self.reward_input_dim = self.obs_feature_dim * self.n_obs_steps + self.z_dim + self.horizon * self.action_dim
        self.reward_output_dim = 1
        self.reward_layer_dims = [self.mlp_hidden_dim, self.mlp_hidden_dim]
        self.reward_network = MLP(
            input_dim=self.reward_input_dim,
            output_dim=self.reward_output_dim,
            layer_dims=self.reward_layer_dims,
        )

    def train_on_batch(self, batch):
        # update cvae
        predict_action, mu, logvar, ntarget_actions = self._forward_training(batch)
        action_loss, kl_loss = self._compute_cvae_loss(predict_action, ntarget_actions, mu, logvar)
        loss = action_loss + kl_loss
        loss_value = loss.item()
        action_loss_value = action_loss.item()
        kl_loss_value = kl_loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.nets['model'].parameters(), 0.001)
        self.optimizer.step()
        return {'loss': loss_value, 'action_loss': action_loss_value, 'KL_loss': kl_loss_value}


    def _forward_training(self, batch):
        obs_dict = deepcopy(batch['obs'])
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][...,:3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        if batch_size < self.train_batch_size:
            # repeat the last batch to match the batch size
            last_point_cloud = nobs['point_cloud'][-1].unsqueeze(0).repeat(self.train_batch_size - batch_size, 1, 1, 1)
            last_agent_pos = nobs['agent_pos'][-1].unsqueeze(0).repeat(self.train_batch_size - batch_size, 1, 1)
            nobs['point_cloud'] = torch.cat([nobs['point_cloud'], last_point_cloud], dim=0)
            nobs['agent_pos'] = torch.cat([nobs['agent_pos'], last_agent_pos], dim=0)
            nactions = torch.cat([nactions, nactions[-1].unsqueeze(0).repeat(self.train_batch_size - batch_size, 1, 1)], dim=0)

            # repeat obs_dict to match the batch size
            last_point_cloud = obs_dict['point_cloud'][-1].unsqueeze(0).repeat(self.train_batch_size - batch_size, 1, 1, 1)
            last_agent_pos = obs_dict['agent_pos'][-1].unsqueeze(0).repeat(self.train_batch_size - batch_size, 1, 1)
            obs_dict['point_cloud'] = torch.cat([obs_dict['point_cloud'], last_point_cloud], dim=0)
            obs_dict['agent_pos'] = torch.cat([obs_dict['agent_pos'], last_agent_pos], dim=0)

        batch_size = nactions.shape[0]
        
        # pc_encoder
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.nets['model']['pc_encoder'](this_nobs) # (B * horizon, 128)
        nobs_features = nobs_features.reshape(batch_size, self.n_obs_steps, self.obs_feature_dim)

        nobs_features = nobs_features.reshape(batch_size, self.obs_feature_dim * self.n_obs_steps)
        nactions = nactions.reshape(batch_size, horizon*self.action_dim)

        # concatenate z and actions
        h = torch.cat([nobs_features, nactions], dim=-1) # (B, n_obs_step*obs_feature_dim + horizon*action_dim)
        h = self.nets['model']['cvae_encoder'](h) # (B, z_dim * 2)
        mu, logvar = torch.split(h, int(0.5*h.shape[-1]), dim=-1)
        z_latent = self.sampling(mu, logvar)
        z = torch.cat([nobs_features.reshape(batch_size, -1), z_latent], dim=-1) # (B , n_obs_steps*obs_feature_dim + z_dim)
        z = z.reshape(1,-1) # (1, B * (n_obs_steps*obs_feature_dim + z_dim))

        if self.use_zero_initial:
            expert_action = torch.zeros(batch_size, horizon, self.action_dim).to(self.device)
        else:
            expert_action = self.expert_policy.predict_action(obs_dict)['action_pred'].reshape(batch_size, horizon, self.action_dim) # Not sure about the shape
            expert_action = self.normalizer['action'].normalize(expert_action)
            # add some noise to expert action
            expert_action = expert_action + torch.randn_like(expert_action) * self.expert_noise

        nactions = nactions.reshape(batch_size, horizon, self.action_dim)

        expert_action = expert_action.unsqueeze(2).repeat(1,1,self.planning_horizon,1)
        actions = expert_action.detach().reshape(1, -1)
        init_actions = deepcopy(actions)
        z = z.contiguous().reshape(1, -1)
        # theseus solver
        actions = th.Vector(tensor=actions, name="actions")
        z = th.Variable(tensor=z, name="z")
        # cost function
        def value_cost_fn(optim_vars, aux_vars):
            temp_actions = optim_vars[0].tensor # (1, B * self.planning_horizon * action dim)
            temp_actions = temp_actions.reshape(batch_size, horizon * self.action_dim)
            temp_actions = torch.clamp(temp_actions, -1, 1,)  # action clipping
            temp_z = aux_vars[0].tensor # (1, B * latent dim)
            temp_z = temp_z.reshape(batch_size, self.n_obs_steps * self.obs_feature_dim + self.z_dim)
            x = torch.cat([temp_z, temp_actions], dim=-1)# (B * horizon (latent_dim + action dim) )
            reward =  self.nets["model"]["reward"](x) #(B, 1)
            G = reward
            err = -G.nan_to_num_(0) + 1000 #+ action_penalty
            err = err.mean(axis=0).reshape(1,1)
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
            # #LUDenseSolver LUCudaSparseSolver CholeskyDenseSolver CholmodSparseSolver BaspachoSparseSolver
            max_iterations = self.traj_opt_num, #self.cfg.max_iterations,
            step_size = self.traj_opt_step, #self.cfg.step_size,
        )
        theseus_optim = th.TheseusLayer(optimizer)
        theseus_optim.to(device=self.device, dtype=torch.float32)
        theseus_inputs = {
        "actions": init_actions,
        "z": z,
        }
        updated_inputs, info = theseus_optim.forward(
            theseus_inputs, optimizer_kwargs={"track_best_solution": True, 
            "verbose": False, "damping": self.damping, "backward_mode": "truncated", "backward_num_iterations": 5,})
        # updated_inputs, info = theseus_optim.forward(
        #     theseus_inputs, optimizer_kwargs={"track_best_solution": True, 
        #     "verbose": False, "damping": 0.05})
        updated_actions = updated_inputs['actions']
        updated_actions = updated_actions.reshape(batch_size, horizon, self.planning_horizon, self.action_dim) # (batch size, horizon, action dim)
        
        predict_actions = updated_actions[:,:,0,:] #+ mean[:,:,0,:]


        return predict_actions, mu, logvar, nactions
    
    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0

    def _compute_cvae_loss(self, predict_action, na_target, mu, logvar):
        actions = predict_action
        loss = 0
        l2_loss =  nn.MSELoss()(actions, na_target)
        KL_loss = -10 * torch.mean(0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)
        # cprint(f"l2_loss: {l2_loss}, KL_loss: {KL_loss}", "green")
        # loss = self.action_loss_weight * l2_loss + KL_loss
        return self.action_loss_weight * l2_loss, KL_loss
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(self, obs_dict):
        expert_obs_dict = deepcopy(obs_dict)
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][...,:3]
        
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]

        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1,*x.shape[2:]))
        nobs_features = self.nets['model']['pc_encoder'](this_nobs)
        nobs_features = nobs_features.reshape(batch_size, self.n_obs_steps, self.obs_feature_dim).reshape(batch_size, self.n_obs_steps * self.obs_feature_dim)
        if batch_size == 1:
            nobs_features = nobs_features.repeat(self.train_batch_size, 1)

        z = torch.randn(self.train_batch_size, self.z_dim).to(self.device)
        z = torch.cat([nobs_features, z], dim=-1)
        z = z.reshape(1, -1)

        expert_action = self.expert_policy.predict_action(expert_obs_dict)['action_pred'].reshape(batch_size, self.horizon, self.action_dim)
        if batch_size == 1:
            expert_action = expert_action.repeat(self.train_batch_size, 1, 1)
        expert_action = self.normalizer['action'].normalize(expert_action)
        expert_action = expert_action.unsqueeze(2).repeat(1,1,self.planning_horizon,1)
        actions = expert_action.detach().reshape(1, -1)
        init_actions = deepcopy(actions)

        # theseus solver
        actions = th.Vector(tensor=actions, name="actions")
        z = th.Variable(tensor=z, name="z")
        # cost function
        def value_cost_fn(optim_vars, aux_vars):
            actions = optim_vars[0].tensor # (1, B * self.planning_horizon * action dim)
            actions = actions.reshape(self.train_batch_size, self.horizon * self.action_dim)
            actions = torch.clamp(actions, -1, 1,)  # action clipping
            z = aux_vars[0].tensor # (1, B * latent dim)
            z = z.reshape(self.train_batch_size, self.n_obs_steps * self.obs_feature_dim + self.z_dim)
            x = torch.cat([z, actions], dim=-1)# (B * horizon (latent_dim + action dim) )
            reward =  self.nets["model"]["reward"](x) #(B, 1)
            G = reward
            err = -G.nan_to_num_(0) + 1000 #+ action_penalty
            err = err.mean(axis=0).reshape(1,1)
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
            # #LUDenseSolver LUCudaSparseSolver CholeskyDenseSolver CholmodSparseSolver BaspachoSparseSolver
            max_iterations = self.traj_opt_num, #self.cfg.max_iterations,
            step_size = self.traj_opt_step, #self.cfg.step_size,
        )
        theseus_optim = th.TheseusLayer(optimizer)
        theseus_optim.to(device=self.device, dtype=torch.float32)
        theseus_inputs = {
        "actions": init_actions,
        "z": z,
        }
        with torch.no_grad():
            updated_inputs, info = theseus_optim.forward(
                theseus_inputs, optimizer_kwargs={"track_best_solution": True, 
                "verbose": False, "damping": self.damping, "backward_mode": "truncated", "backward_num_iterations": 5,})
        # updated_inputs, info = theseus_optim.forward(
        #     theseus_inputs, optimizer_kwargs={"track_best_solution": True, 
        #     "verbose": False, "damping": 0.05})
        updated_actions = updated_inputs['actions']
        updated_actions = updated_actions.reshape(self.train_batch_size, self.horizon, self.planning_horizon, self.action_dim) # (batch size, horizon, 1(planning_horizon), action dim)
        predict_actions = updated_actions[:,:,0,:] #+ mean[:,:,0,:]

        if batch_size == 1:
            predict_actions = predict_actions[0].unsqueeze(0)

        predict_actions = self.normalizer['action'].unnormalize(predict_actions)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = predict_actions[:, start:end]

        result = {
            'action': action,
            'action_pred': predict_actions,
        }
        return result
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
        
        

