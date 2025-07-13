import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import random
from collections import deque

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

# Define a simple 10x10 gridworld environment
class SparseGridworld(gym.Env):
    def _init_(self):
        super(SparseGridworld, self)._init_()
        self.grid_size = 10
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=9, shape=(2,), dtype=np.int32)
        self.state = None
        self.goal = (9, 9)
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        self.state = (0, 0)
        self.current_step = 0
        return np.array(self.state)

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.grid_size - 1, y + 1)
        self.state = (x, y)
        self.current_step += 1

        reward = 1.0 if self.state == self.goal else 0.0
        done = self.state == self.goal or self.current_step >= self.max_steps
        return np.array(self.state), reward, done, {}

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        x, y = self.state
        grid[x, y] = 1
        gx, gy = self.goal
        grid[gx, gy] = 2
        print(grid)

# Intrinsic Curiosity Module (ICM)
class ICM(nn.Module):
    def _init_(self, state_dim, action_dim, hidden_dim=64):
        super(ICM, self)._init_()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.inverse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, state, next_state, action):
        state_f = self.feature(state)
        next_state_f = self.feature(next_state)
        pred_action = self.inverse(torch.cat([state_f, next_state_f], dim=-1))
        pred_next_f = self.forward_model(torch.cat([state_f, action], dim=-1))
        return pred_next_f, pred_action

# Simplified PPO Policy
class PPOPolicy(nn.Module):
    def _init_(self, state_dim, action_dim, hidden_dim=64):
        super(PPOPolicy, self)._init_()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

# Training loop
def train_option_discovery():
    env = SparseGridworld()
    state_dim = 2
    action_dim = 4
    icm = ICM(state_dim, action_dim)
    policy = PPOPolicy(state_dim, action_dim)
    icm_optimizer = optim.Adam(icm.parameters(), lr=0.001)
    policy_optimizer = optim.Adam(policy.parameters(), lr=0.001)
    buffer = deque(maxlen=10000)
    lambda_intrinsic = 0.1
    episodes = 1000
    trajectories = []

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).clone()
        episode_traj = []
        total_reward = 0
        done = False
        step = 0

        while not done and step < env.max_steps:
            probs = policy(state)
            action = torch.multinomial(probs, 1).item()
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).clone()
            episode_traj.append((state.clone(), action, next_state.clone()))

            # Compute intrinsic reward and detach to break computation graph
            action_onehot = torch.zeros(1, action_dim)
            action_onehot[0, action] = 1
            pred_next_f, pred_action = icm(state, next_state, action_onehot)
            intrinsic_reward = torch.norm(pred_next_f - icm.feature(next_state), dim=-1).pow(2).detach()
            total_reward += reward + lambda_intrinsic * intrinsic_reward.item()

            # Store transition
            buffer.append((state.clone(), action, reward, next_state.clone(), intrinsic_reward.clone()))

            # Update models
            if len(buffer) > 32:
                batch = random.sample(buffer, 32)
                states, actions, rewards, next_states, intrinsic_rewards = zip(*batch)
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                actions = torch.tensor(actions)
                action_onehot = torch.zeros(len(actions), action_dim)
                action_onehot.scatter_(1, actions.unsqueeze(1), 1)
                intrinsic_rewards = torch.cat(intrinsic_rewards)

                # Policy loss (simplified PPO)
                policy_optimizer.zero_grad()
                probs = policy(states)
                log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)))
                policy_loss = -(log_probs * (torch.tensor(rewards) + lambda_intrinsic * intrinsic_rewards)).mean()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                policy_optimizer.step()

                # ICM loss
                icm_optimizer.zero_grad()
                pred_next_f, pred_action = icm(states, next_states, action_onehot)
                icm_loss = torch.norm(pred_next_f - icm.feature(next_states), dim=-1).pow(2).mean() + \
                           nn.CrossEntropyLoss()(pred_action, actions)
                icm_loss.backward()
                torch.nn.utils.clip_grad_norm_(icm.parameters(), max_norm=1.0)
                icm_optimizer.step()

            state = next_state
            step += 1

        trajectories.append(episode_traj)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    # Cluster trajectories using k-means (simplified PCCA+)
    states = np.concatenate([np.array([s.numpy() for s, _, _ in traj]) for traj in trajectories])
    kmeans = KMeans(n_clusters=10)
    option_labels = kmeans.fit_predict(states)
    print("Option labels:", np.unique(option_labels, return_counts=True))

    # Evaluate state coverage
    unique_states = np.unique(states, axis=0)
    coverage = len(unique_states) / (env.grid_size ** 2) * 100
    print(f"State coverage: {coverage}%")

if _name_ == "_main_":
    train_option_discovery()
