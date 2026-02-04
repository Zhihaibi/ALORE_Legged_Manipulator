# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-B2Z1MultiObjWBCGNNPLAN-Direct-v0",
    entry_point=f"{__name__}.b2z1_multiobj_wbc_gnn_plan_env:B2Z1MultiObjWBCGNNPLANEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2z1_multiobj_wbc_gnn_plan_env:B2Z1MultiObjWBCGNNPLANFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2Z1MultiObjWBCGNNPLANFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-B2Z1MultiObjWBCGNNPLAN-Direct-v0",
    entry_point=f"{__name__}.b2z1_multiobj_wbc_gnn_plan_env:B2Z1MultiObjWBCGNNPLANEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2z1_multiobj_wbc_gnn_plan_env:B2Z1MultiObjWBCGNNPLANRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2Z1MultiObjWBCGNNPLANRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
