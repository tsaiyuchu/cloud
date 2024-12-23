import gymnasium as gym
import highway_env
from stable_baselines3 import DQN

# 创建 intersection-v0 环境
env = gym.make("intersection-v0")

# 配置环境参数（可根据需要进行调整）
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted"
    },
    "simulation_frequency": 15,  # 仿真频率 [Hz]
    "policy_frequency": 5,       # 策略更新频率 [Hz]
    "duration": 50,              # 每个 episode 的持续时间
    "collision_reward": -1,      # 碰撞惩罚
    "reward_speed_range": [0, 30],  # 奖励速度范围
    "normalize_reward": True     # 归一化奖励
}
env.configure(config)

# 创建 DQN 模型
model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=1,
    tensorboard_log="./dqn_intersection_tensorboard/"
)

# 训练模型
model.learn(total_timesteps=100000)

# 保存模型
model.save("dqn_intersection_model")

# 测试训练好的模型
env = gym.make("intersection-v0", render_mode="human")
obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
