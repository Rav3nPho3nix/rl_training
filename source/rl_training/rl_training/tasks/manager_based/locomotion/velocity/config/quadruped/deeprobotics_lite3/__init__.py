# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Flat-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLite3FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsLite3FlatTrainerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsLite3RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsLite3RoughTrainerCfg",
    },
)

gym.register(
    id="Custom-Lite3-Long-Jump-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft",
    },
)

gym.register(
    id="Custom-Lite3-Long-Jump-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_1",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_1",
    },
)

gym.register(
    id="Custom-Lite3-Long-Jump-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_2",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_2",
    },
)

gym.register(
    id="Custom-Lite3-Long-Jump-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_3",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_3",
    },
)

gym.register(
    id="Custom-Lite3-Long-Jump-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_4",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_4",
    },
)

gym.register(
    id="Custom-Lite3-Long-Jump-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_5",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_5",
    },
)

gym.register(
    id="Custom-Lite3-Long-Jump-v6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_6",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_6",
    },
)

gym.register(
    id="Custom-Lite3-Long-Jump-v7",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_7",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_7",
    },
)

gym.register(
    id="Custom-Lite3-Long-Jump-v8",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_8",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_8",
    },
)

gym.register(
    id="Custom-Lite3-Long-Jump-v9",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_9",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_9",
    },
)


gym.register(
    id="Custom-Lite3-Long-Jump-v10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_10",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_10",
    },
)


gym.register(
    id="Custom-Lite3-Long-Jump-v11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_11",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_11",
    },
)

gym.register(
    id="Custom-Lite3-Long-Jump-v12",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_jump_env_cfg:CustomLite3LongJumpEnvCfg_12",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3LongJumpPPORunnerCft_12",
    },
)

gym.register(
    id="Custom-Lite3-Rear-Balance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rear_balance_env_cfg:CustomLite3RearBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3RearBalancePPORunnerCft",
    },
)

gym.register(
    id="Custom-Lite3-Rear-Balance-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rear_balance_env_cfg:CustomLite3RearBalanceEnvCfg_1",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3RearBalancePPORunnerCft_1",
    },
)

gym.register(
    id="Custom-Lite3-Rear-Balance-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rear_balance_env_cfg:CustomLite3RearBalanceEnvCfg_2",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3RearBalancePPORunnerCft_2",
    },
)

gym.register(
    id="Custom-Lite3-Rear-Balance-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rear_balance_env_cfg:CustomLite3RearBalanceEnvCfg_3",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:CustomLite3RearBalancePPORunnerCft_3",
    },
)