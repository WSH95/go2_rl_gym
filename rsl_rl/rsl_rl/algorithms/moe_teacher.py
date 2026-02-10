# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage


class MoETeacher(PPO):
    """PPO variant for teacher-only MoE policies.

    The actor consumes privileged observations. To keep PPO.update() unchanged,
    privileged observations are stored in both actor and critic rollout buffers.
    """

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            critic_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        self.transition.actions = self.actor_critic.act(obs, privileged_obs=critic_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # Keep both storages aligned to privileged inputs for PPO.update().
        self.transition.observations = critic_obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
