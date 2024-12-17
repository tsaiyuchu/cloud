import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Define the Neural Network for Q-value approximation
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# DQN Agent Class
class DQNAgent:
    def __init__(self, env, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Q-Network and Target Network
        self.q_network = DQNNetwork(self.state_dim, self.action_dim)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.memory = deque(maxlen=10000)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Calculate target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions).squeeze()

        # Loss and optimization
        loss = nn.MSELoss()(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_dqn(agent, env, num_episodes=500, max_steps=200, batch_size=64, target_update=10):
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

            agent.replay(batch_size)

        agent.adjust_epsilon()
        rewards.append(total_reward)

        if episode % target_update == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}")

    return rewards

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env)

    rewards = run_dqn(agent, env)
    env.close()

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CartPole DQN')
    plt.show()
