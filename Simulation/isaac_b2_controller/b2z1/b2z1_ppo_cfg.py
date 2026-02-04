# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class B2Z1MultiObjWBCGNNPLANFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # logger = "wandb"
    # wandb_project = "b2z1_multiobj_wbc_gnn"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "b2z1_multiobj_wbc_gnn_plan_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="PhysicActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name= "PhysicPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class B2Z1MultiObjWBCGNNPLANRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # logger = "tensorboard"
    # logger = "wandb"
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "b2z1_multiobj_wbc_gnn_plan_rough_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
