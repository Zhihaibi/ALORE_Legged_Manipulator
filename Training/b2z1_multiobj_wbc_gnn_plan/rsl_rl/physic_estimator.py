import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PhysicEstimator(nn.Module):
    def __init__(self,
                 input_dim=44,
                 output_dim=3,  # [mass, mu]
                 lstm_hidden_size=128,
                 lstm_layers=1,
                 mlp_hidden_dim=64,
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actor_obs = input_dim  # Number of observations used for one-step prediction
        self.history_length = 11  # Assuming the history length is 11

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_layers,
            batch_first = True
        )

        # Output head: MLP
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )

        # Optimizer and loss
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.max_grad_norm = max_grad_norm

        self.to(self.device)

        self.num_one_step_obs = input_dim  # Number of observations used for one-step prediction
        
        print(f"PhysicEstimator initialized with input_dim={input_dim}, output_dim={output_dim}, ")


    def forward(self, obs_history):
        """
        obs_history: (B, T, D)
        Returns: (B, 2)
        """
        B = obs_history.shape[0]
        T = self.history_length
        D = self.num_actor_obs

        obs_history = obs_history.view(B, T, D)  # 💡 关键 reshape

        # print(f"====obs_history shape: {obs_history.shape}====")
        lstm_out, (h_n, _) = self.lstm(obs_history)  # h_n: (num_layers, B, H)
        # print(f"====h_n shape: {h_n.shape}====")
        last_hidden = h_n[-1]  # (B, H)
        # print(f"====last_hidden shape: {last_hidden.shape}====")
        return self.output_head(last_hidden)
    

    def update(self, obs_history, critic_obs):
        """
        obs_history: (B, T, D)
        mass_gt, mu_gt: (B,) or (B, 1)
        """
        # print(f"====PhysicEstimator update called====")

        obj_ang_vel_z_gt = critic_obs[:, -4].detach()
        obj_lin_vel_x_gt = critic_obs[:, -9].detach()
        obj_lin_vel_y_gt = critic_obs[:, -8].detach()
        x = obs_history  # (B, T, D)
        y = torch.stack([obj_lin_vel_x_gt, obj_lin_vel_y_gt, obj_ang_vel_z_gt], dim=-1)  # (B, 2)

        self.train()
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()


    def predict(self, obs_history):
        """
        obs_history: (B, T, D)
        Returns: (B, 2) as numpy array
        """
        if not isinstance(obs_history, torch.Tensor):
            obs_history = torch.tensor(obs_history, dtype=torch.float32)

        x = obs_history.to(self.device)

        self.eval()
        with torch.no_grad():
            y_pred = self.forward(x)

        return y_pred.cpu().numpy()
