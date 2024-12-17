import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def get_discrete_state(state, bins, lower_bounds, upper_bounds):
    ratios = [(state[i] - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_state = [int(np.digitize(ratios[i], bins[i]) - 1) for i in range(len(state))]
    new_state = [min(len(bins[i]) - 1, max(0, new_state[i])) for i in range(len(state))]
    return tuple(new_state)

class Agent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.n_bins = [6, 6, 6, 6] 
        self.lower_bounds = [env.observation_space.low[0], -3.0, env.observation_space.low[2], -3.5]
        self.upper_bounds = [env.observation_space.high[0], 3.0, env.observation_space.high[2], 3.5]
        self.bins = [np.linspace(self.lower_bounds[i], self.upper_bounds[i], self.n_bins[i] - 1) for i in range(len(self.n_bins))]

        self.q_table = np.random.uniform(low=0, high=1, size=(*self.n_bins, env.action_space.n))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_qlearning(agent, env, num_episodes=20, max_steps=200):
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = get_discrete_state(state, agent.bins, agent.lower_bounds, agent.upper_bounds)

        total_reward = 0

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = get_discrete_state(next_state, agent.bins, agent.lower_bounds, agent.upper_bounds)
            agent.learn(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        agent.adjust_epsilon()
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    return rewards

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    agent = Agent(env)
    rewards = run_qlearning(agent, env)
    env.close()

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward_sum')
    plt.title('CartPole Q-learn')
    plt.show()
