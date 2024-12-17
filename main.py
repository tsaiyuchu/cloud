import gymnasium as gym

n_episodes = 10  # 總共玩幾次
episode_length = 20  # 每個回合的最大步數

# 環境建立
env = gym.make("CartPole-v1", render_mode="human")

for episode in range(n_episodes):
    observation, info = env.reset()
    reward_sum = 0

    for t in range(episode_length):
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        if terminated or truncated:
            print(f"Episode {episode + 1} finished after {t + 1} timesteps with reward {reward_sum}")
            break

env.close()
