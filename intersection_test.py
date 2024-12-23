import gymnasium as gym
import highway_env
from stable_baselines3 import DQN

# 加载训练好的模型
model = DQN.load("C:/Users/user/Desktop/test/dqn_intersection_model.zip")

# 创建环境，设置渲染模式为 'human'
env = gym.make("intersection-v0", render_mode="human")

# 运行测试
num_episodes = 20
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        # 渲染当前帧
        env.render()
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# 关闭环境
env.close()
