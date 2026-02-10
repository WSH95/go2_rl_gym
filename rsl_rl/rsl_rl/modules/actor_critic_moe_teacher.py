# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.utils import MLP, MoE


class ActorCriticMoETeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        expert_num=8,
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticMoETeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        # Teacher actor uses privileged observations.
        self.actor = MoE(
            expert_num=expert_num,
            input_dim=num_critic_obs,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation,
        )

        # Critic keeps standard PPO-style value regression on privileged observations.
        self.critic = MLP([num_critic_obs, *critic_hidden_dims, 1], activation=activation)

        print(f"Actor MoE: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _get_actor_input(self, observations, privileged_obs=None):
        if privileged_obs is not None:
            return privileged_obs
        if observations.shape[-1] == self.num_critic_obs:
            return observations
        if observations.shape[-1] == self.num_actor_obs and self.num_critic_obs > self.num_actor_obs:
            pad_shape = list(observations.shape)
            pad_shape[-1] = self.num_critic_obs - self.num_actor_obs
            padding = observations.new_zeros(pad_shape)
            return torch.cat([observations, padding], dim=-1)
        raise RuntimeError(
            f"Invalid actor input shape {observations.shape[-1]} for ActorCriticMoETeacher "
            f"(num_actor_obs={self.num_actor_obs}, num_critic_obs={self.num_critic_obs})."
        )

    def update_distribution(self, observations, privileged_obs=None):
        actor_input = self._get_actor_input(observations, privileged_obs=privileged_obs)
        actor_output = self.actor(actor_input)
        mean = actor_output[0] if isinstance(actor_output, (tuple, list)) else actor_output
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, privileged_obs=None, **kwargs):
        self.update_distribution(observations, privileged_obs=privileged_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actor_input = self._get_actor_input(observations, privileged_obs=None)
        actor_output = self.actor(actor_input)
        actions_mean = actor_output[0] if isinstance(actor_output, (tuple, list)) else actor_output
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
