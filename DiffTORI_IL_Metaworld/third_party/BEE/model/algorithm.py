import os
import numpy as np
import ipdb
import torch
import copy
import torch.nn.functional as F
from torch.optim import Adam
from .utils import soft_update, hard_update
from .model import GaussianPolicy, ValueNetwork, QNetwork, DeterministicPolicy


class BAC(object):
    def __init__(self, num_inputs, action_space, args):


        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.quantile = args.quantile
        self.lambda_method = args["lambda"]
        self._max_q_grad = 1e-7

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda:{}".format(str(args.device)) if args.cuda else "cpu")
        self._last_q_grad = torch.ones((args.batch_size, 1)).to(device=self.device)

        self.Q_critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.Q_critic_optim = Adam(self.Q_critic.parameters(), lr=args.lr)

        self.V_critic = ValueNetwork(num_inputs, args.hidden_size).to(device=self.device)
        self.V_critic_optim = Adam(self.V_critic.parameters(), lr=args.lr)

        self.Q_critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.Q_critic_target, self.Q_critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]


    def save_checkpoint(self, path, i_episode):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.Q_critic.state_dict(),
                    'critic_target_state_dict': self.Q_critic_target.state_dict(),
                    'value_state_dict': self.V_critic.state_dict(),
                    'critic_optimizer_state_dict': self.Q_critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                    'value_optimizer_state_dict': self.V_critic_optim.state_dict()}, ckpt_path)
    
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.Q_critic.load_state_dict(checkpoint['critic_state_dict'])
            self.Q_critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.V_critic.load_state_dict(checkpoint['value_state_dict'])
            self.Q_critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.V_critic_optim.load_state_dict(checkpoint['value_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.Q_critic.eval()
                self.Q_critic_target.eval()
                self.V_critic.eval()
            else:
                self.policy.train()
                self.Q_critic.train()
                self.Q_critic_target.train()
                self.V_critic.train()
