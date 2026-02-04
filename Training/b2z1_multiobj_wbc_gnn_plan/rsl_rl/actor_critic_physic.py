# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import csv
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from .physic_estimator import PhysicEstimator
from rsl_rl.utils import resolve_nn_activation
from .interactive_gnn import InteractiveGNN 


class PhysicActorCritic(ActorCritic):

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        # 11*(44+2)
        # print(f"====PhysicActorCritic input dimensions: actor={num_actor_obs}, critic={num_critic_obs}, actions={num_actions}====")
        self.history_length = 11
 
        super().__init__(
            num_actor_obs=num_actor_obs + self.history_length*3+ 128, #TODO： hard code, 11*3 for velocity prediction ## TODO: with only physical estimation
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs
        )

        #
        if not hasattr(self, 'device'):
            self.device = kwargs.get('device', 'cpu')

        self.num_actor_obs = int(num_actor_obs / self.history_length)

        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs + self.history_length*3 + 128 # 11*3 for velocity prediction, 128 for GNN output ## TODO: with only physical estimation
        mlp_input_dim_c = num_critic_obs

        # multi-head actor policy
        shared_layers = []
        shared_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        shared_layers.append(activation)
        for i in range(len(actor_hidden_dims) - 1):
            shared_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i+1]))
            shared_layers.append(activation)
        self.shared_mlp = nn.Sequential(*shared_layers)

        # base control 
        self.base_head = nn.Linear(actor_hidden_dims[-1], 3)
        # arm control 
        self.arm_head = nn.Linear(actor_hidden_dims[-1], 6)

        # Add a physic estimator
        self.physic_estimator = PhysicEstimator(
            input_dim = self.num_actor_obs,  # Assuming actor obs is used for estimation
            output_dim=3,  # [vx, vy, omega] ## TODO: with only physical estimation
            device=self.device
        )
        print(f'Estimator: {self.physic_estimator}')

        ## add the interactive GNN
        self.interactive_gnn = InteractiveGNN(
            node_dim=15, # node feature dimension
            edge_dim=7,  # edge feature dimension
            hidden_dim=64,
            out_dim=128
        )

        self.num_one_step_obs = num_actor_obs  # Number of observations used for one-step prediction



    def update_distribution(self, observations, critic_observations):
        # velocity prediction
        with torch.no_grad():
            physic_estimator = self.physic_estimator(observations)

        B = observations.shape[0]
        T = self.history_length
        D = self.num_actor_obs
        obs_seq = observations.view(B, T, D)  # (B, T, D)

        obj_ang_vel_z_gt = critic_observations[:, -4].view(B, 1, 1).expand(-1, T, -1) # (B, T, 1)
        obj_lin_vel_x_gt = critic_observations[:, -9].view(B, 1, 1).expand(-1, T, -1)
        obj_lin_vel_y_gt = critic_observations[:, -8].view(B, 1, 1).expand(-1, T, -1)

        lin_vel_x_pre = physic_estimator[:, :1]
        lin_vel_y_pre = physic_estimator[:, 1:2]
        ang_vel_z_pre = physic_estimator[:, 2:3]
        lin_vel_x_pre = lin_vel_x_pre.unsqueeze(1).expand(-1, T, -1)  # (B, T, 1)
        lin_vel_y_pre = lin_vel_y_pre.unsqueeze(1).expand(-1, T, -1)  # (B, T, 1)
        ang_vel_z_pre = ang_vel_z_pre.unsqueeze(1).expand(-1, T, -1)  # (B, T, 1)

        obs_augmented = torch.cat((obs_seq, lin_vel_x_pre, lin_vel_y_pre, ang_vel_z_pre), dim=-1)  

        ## interactive GNN processing
        node_features, edge_index, edge_attr, batch = self.interactive_gnn.build_interaction_graph(obs_seq, critic_observations)
        z = self.interactive_gnn(node_features, edge_index, edge_attr, batch)  # shape: [B, 128]

        actor_input = torch.cat([obs_augmented.reshape(B, -1), z], dim=-1)  # (B, 634)

        # actor_input = obs_augmented.reshape(B, -1)  # (B, 506)

        shared_feat = self.shared_mlp(actor_input)
        base_mean = self.base_head(shared_feat)
        arm_mean = self.arm_head(shared_feat)
        mean = torch.cat([base_mean, arm_mean], dim=-1)

        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)


    def act(self, observations, critic_observations, **kwargs):
        self.update_distribution(observations, critic_observations)
        return self.distribution.sample()


    def act_inference(self, observations, critic_observations):
        # velocity prediction
        physic_estimator = self.physic_estimator(observations)

        B = observations.shape[0]
        T = self.history_length
        D = self.num_actor_obs
        obs_seq = observations.view(B, T, D)  # (B, T, D)

        obj_lin_vel_x_pre = physic_estimator[:, :1]  
        obj_lin_vel_y_pre = physic_estimator[:, 1:2]  
        obj_lin_vel_z_pre = physic_estimator[:, 2:3]
        

        # print("plan_vel predict", obj_lin_vel_x_pre/2., obj_lin_vel_y_pre/2., obj_lin_vel_z_pre*4.0)

        obj_ang_vel_z_gt = critic_observations[:, -4].view(B, 1, 1).expand(-1, T, -1) # (B, T, 1)
        obj_lin_vel_x_gt = critic_observations[:, -9].view(B, 1, 1).expand(-1, T, -1)
        obj_lin_vel_y_gt = critic_observations[:, -8].view(B, 1, 1).expand(-1, T, -1)
    

        obj_ang_vel_z_gt = critic_observations[:, -4]  # (B,)
        obj_lin_vel_x_gt = critic_observations[:, -9]  # (B,)
        obj_lin_vel_y_gt = critic_observations[:, -8]  # (B,)

        # print("obj_ang_vel_xyz_gt", obj_lin_vel_x_gt/2, obj_lin_vel_y_gt/2, obj_ang_vel_z_gt*4.0)


        obj_lin_vel_x_pre_s = physic_estimator[:, :1].flatten()   # (B, 1) -> (B,)
        obj_lin_vel_y_pre_s = physic_estimator[:, 1:2].flatten()  # (B, 1) -> (B,)
        obj_lin_vel_z_pre_s = physic_estimator[:, 2:3].flatten()  # (B, 1) -> (B,)
        # print("obj_lin_vel_x_pre_s", obj_lin_vel_x_pre_s[0])


         
        # self._save_predictions_and_gt_to_csv(obj_lin_vel_x_pre_s/2, obj_lin_vel_y_pre_s/2, obj_lin_vel_z_pre_s*4.0, 
        #                                  obj_lin_vel_x_gt/2, obj_lin_vel_y_gt/2, obj_ang_vel_z_gt*4.0)
    

        obj_lin_vel_x_pre = obj_lin_vel_x_pre.unsqueeze(1).expand(-1, T, -1)  # (B, T, 1)
        obj_lin_vel_y_pre = obj_lin_vel_y_pre.unsqueeze(1).expand(-1, T, -1)  # (B, T, 1)
        obj_lin_vel_z_pre = obj_lin_vel_z_pre.unsqueeze(1).expand(-1, T, -1)  # (B, T, 1)
        obs_augmented = torch.cat((obs_seq, obj_lin_vel_x_pre, obj_lin_vel_y_pre, obj_lin_vel_z_pre), dim=-1)  # TODO: (B, T, 46)


        # obj_lin_vel_x_pre_zero = torch.zeros_like(obj_lin_vel_x_pre)  # (B, T, 1)
        # obj_lin_vel_y_pre_zero = torch.zeros_like(obj_lin_vel_y_pre)  # (B, T, 1)
        # obj_lin_vel_z_pre_zero = torch.zeros_like(obj_lin_vel_z_pre)  # (B, T, 1)
        # obs_augmented = torch.cat((obs_seq, obj_lin_vel_x_pre_zero, obj_lin_vel_y_pre_zero, obj_lin_vel_z_pre_zero), dim=-1)  # TODO: (B, T, 46)


        # interactive GNN processing
        node_features, edge_index, edge_attr, batch = self.interactive_gnn.build_interaction_graph(obs_seq, critic_observations)
        z = self.interactive_gnn(node_features, edge_index, edge_attr, batch)  # shape: [B, 128]

        # if hasattr(self, 'visualization_counter'):
        #     self.visualization_counter += 1
        # else:
        #     self.visualization_counter = 0

        # if self.visualization_counter % 200 == 0:  
        #     try:
        #         env_labels = torch.arange(z.shape[0]) // (z.shape[0] // 3)
                
        #         if z.shape[0] >= 5:
        #             
        #            results = self.visualize_gnn_features_pca_only(
        #                 z, labels=env_labels.cpu().numpy(), 
        #                 crop_mode='break',  
        #                 # break_x=((0.5, 1.5),), 
        #                 # break_y=((-0.2, 0.2),),
        #                 save_path=f'gnn_pca_broken_step_{self.visualization_counter}.png'
        #             )
                    
        #            
        #             # if results:
        #             #     np.savez(f'gnn_pca_analysis_step_{self.visualization_counter}.npz', **results)
        #         else:
        #             print(f"Skipping visualization: only {z.shape[0]} samples")
                    
        #     except Exception as e:
        #         print(f"PCA visualization failed: {e}")
 
        
        actor_input = torch.cat([obs_augmented.reshape(B, -1), z], dim=-1)  # (B, 634)

        # actor_input = obs_augmented.reshape(B, -1)  # (B, 506)

        shared_feat = self.shared_mlp(actor_input)
        base_mean = self.base_head(shared_feat)
        arm_mean = self.arm_head(shared_feat)
        actions_mean = torch.cat([base_mean, arm_mean], dim=-1)

        return actions_mean
    

   
    def _save_predictions_and_gt_to_csv(self, pred_vx, pred_vy, pred_omega, 
                                    gt_vx, gt_vy, gt_omega):

        csv_file = "plan_vel_predictions_vs_gt_table_square.csv"
        
        # 
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # 
            if not file_exists:
                writer.writerow([
                    'timestamp', 'env_id', 
                    'pred_vx', 'pred_vy', 'pred_omega',
                    'gt_vx', 'gt_vy', 'gt_omega',
                    'error_vx', 'error_vy', 'error_omega'
                ])
            
         
            if hasattr(pred_vx, 'cpu'):
                pred_vx = pred_vx.cpu().numpy()
                pred_vy = pred_vy.cpu().numpy()
                pred_omega = pred_omega.cpu().numpy()
                gt_vx = gt_vx.cpu().numpy()
                gt_vy = gt_vy.cpu().numpy()
                gt_omega = gt_omega.cpu().numpy()
            
        
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            for env_id in range(len(pred_vx)):
   
                error_vx = pred_vx[env_id] - gt_vx[env_id]
                error_vy = pred_vy[env_id] - gt_vy[env_id]
                error_omega = pred_omega[env_id] - gt_omega[env_id]
                
                writer.writerow([
                    current_time,
                    env_id,
                    pred_vx[env_id],
                    pred_vy[env_id], 
                    pred_omega[env_id],
                    gt_vx[env_id],
                    gt_vy[env_id],
                    gt_omega[env_id],
                    error_vx,
                    error_vy,
                    error_omega
                ])


    
    def visualize_gnn_features_pca_only(
        self,
        z,
        labels=None,
        save_path='gnn_pca_visualization.png',
        scale_mode='none',          
        crop_mode='none',           # 
        crop_lo=2, crop_hi=98,      
        margin_ratio=0.06,          
        symlog_linthresh=1e-3,      
        break_x=((0.0, 0.3),),      
        break_y=((-0.05, 0.0),)     
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from scipy.stats import chi2
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # --- ---
        z_np = z.cpu().detach().numpy() if hasattr(z, 'cpu') else np.array(z)
        if z_np.shape[0] < 5:
            print(f"Too few samples ({z_np.shape[0]}) for visualization")
            return

        # --- ---
        def draw_confidence_ellipse(ax, x, y, color, alpha=0.3, confidence=0.95):
            if len(x) < 3: 
                return
            try:
                mean = np.array([np.mean(x), np.mean(y)])
                cov  = np.cov(x, y)
                eigvals, eigvecs = np.linalg.eigh(cov + 1e-12*np.eye(2))
                order = np.argsort(eigvals)[::-1]
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                chi2_val = chi2.ppf(confidence, df=2)
                width  = 2*np.sqrt(chi2_val*eigvals[0])
                height = 2*np.sqrt(chi2_val*eigvals[1])
                angle  = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
                ell = Ellipse(mean, width, height, angle=angle,
                            facecolor=color, edgecolor=color, linewidth=2, alpha=alpha)
                ax.add_patch(ell)
            except Exception as e:
                print(f"Failed to draw ellipse: {e}")

        # ------
        pca = PCA(n_components=2, random_state=42)
        z_pca = pca.fit_transform(z_np)
        explained_var = pca.explained_variance_ratio_

        z_plot = z_pca.copy()
        if scale_mode == 'zscore':
            z_plot = StandardScaler().fit_transform(z_plot)

        print(f"PCA data range: X[{z_plot[:, 0].min():.3f}, {z_plot[:, 0].max():.3f}], Y[{z_plot[:, 1].min():.3f}, {z_plot[:, 1].max():.3f}]")

        object_names = ['Table1', 'Table2', 'Chair']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # ----
        if crop_mode == 'break':
            try:
                from brokenaxes import brokenaxes
                
     
                x_min, x_max = z_plot[:, 0].min(), z_plot[:, 0].max()
                y_min, y_max = z_plot[:, 1].min(), z_plot[:, 1].max()
                
                print(f"Setting up broken axes with data range: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}]")
                

                xlims = [(x_min - 0.1, break_x[0][0]), (break_x[0][1], x_max + 0.1)]
                ylims = [(y_min - 0.1, break_y[0][0]), (break_y[0][1], y_max + 0.1)]
                
                fig = plt.figure(figsize=(12, 10))
                bax = brokenaxes(
                    xlims=xlims,
                    ylims=ylims,
                    hspace=0.05, 
                    wspace=0.05,
                    despine=False
                )
                scatter_ax = bax
                
                print("Broken axes created successfully")
                
            except ImportError:
                print("brokenaxes not available, falling back to regular plot")
                crop_mode = 'none'
                plt.figure(figsize=(10, 8))
                scatter_ax = plt.gca()
            except Exception as e:
                print(f"Failed to create broken axes: {e}, falling back to regular plot")
                crop_mode = 'none'
                plt.figure(figsize=(10, 8))
                scatter_ax = plt.gca()
        else:
            plt.figure(figsize=(10, 8))
            scatter_ax = plt.gca()


        if labels is not None:
            unique_labels = np.unique(labels)
            for i, lab in enumerate(unique_labels):
                m = (labels == lab)
                color = colors[i % len(colors)]

                scatter_ax.scatter(z_plot[m, 0], z_plot[m, 1],
                        c=color, label=object_names[i % len(object_names)],
                        alpha=0.85, s=80, edgecolors='black', linewidth=0.6)
                
                if np.sum(m) >= 3:
                    draw_confidence_ellipse(scatter_ax, z_plot[m, 0], z_plot[m, 1], color)
                    
                    cx, cy = z_plot[m, 0].mean(), z_plot[m, 1].mean()
                    scatter_ax.scatter(cx, cy, c='white', s=120, edgecolors=color,
                            linewidth=2, marker='x', zorder=10)
        else:
            scatter_ax.scatter(z_plot[:, 0], z_plot[:, 1],
                            alpha=0.8, s=70, edgecolors='black', linewidth=0.4)

        if crop_mode == 'percentile':
            xlo, xhi = np.percentile(z_plot[:, 0], [crop_lo, crop_hi])
            ylo, yhi = np.percentile(z_plot[:, 1], [crop_lo, crop_hi])
            xr, yr = xhi - xlo, yhi - ylo
            scatter_ax.set_xlim(xlo - xr*margin_ratio, xhi + xr*margin_ratio)
            scatter_ax.set_ylim(ylo - yr*margin_ratio, yhi + yr*margin_ratio)
        elif crop_mode == 'tight':
            xlo, xhi = z_plot[:, 0].min(), z_plot[:, 0].max()
            ylo, yhi = z_plot[:, 1].min(), z_plot[:, 1].max()
            xr, yr = xhi - xlo, yhi - ylo
            scatter_ax.set_xlim(xlo - xr*margin_ratio, xhi + xr*margin_ratio)
            scatter_ax.set_ylim(ylo - yr*margin_ratio, yhi + yr*margin_ratio)

        if scale_mode == 'symlog':
            try:
                scatter_ax.set_xscale('symlog', linthresh=symlog_linthresh)
                scatter_ax.set_yscale('symlog', linthresh=symlog_linthresh)
            except:
                print("symlog scale not supported with broken axes")

        if crop_mode != 'break':   
            scatter_ax.set_title(
                'GNN Feature PCA Visualization with 95% Confidence Ellipses\n'
                f'Explained Variance: {explained_var.sum():.3f}',
                fontsize=14, fontweight='bold', pad=18
            )
            scatter_ax.set_xlabel(f'PC1 ({explained_var[0]:.3f})', fontsize=12)
            scatter_ax.set_ylabel(f'PC2 ({explained_var[1]:.3f})', fontsize=12)
            scatter_ax.grid(True, alpha=0.3)
        else:
            fig.suptitle(
                'GNN Feature PCA Visualization with 95% Confidence Ellipses\n'
                f'Explained Variance: {explained_var.sum():.3f}',
                fontsize=14, fontweight='bold', y=0.95
            )

        if labels is not None:
            if crop_mode != 'break':
                scatter_ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=False)
            else:
                fig.legend(labels=[object_names[i] for i in range(len(np.unique(labels)))], 
                          loc='upper right', fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"PCA visualization saved to {save_path}")

        return {'pca_components': z_pca, 'explained_variance': explained_var}
