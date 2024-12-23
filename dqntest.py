import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import torch
import highway_env
import time

def main():
    # 檢查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train = False
    model_path = "highway_dqn/model_optimized"

    if train:
        # 建立訓練環境，並設定 render_mode 以便於觀察
        env = gym.make("intersection-v0", render_mode="human")

        # 建立 DQN 模型
        model = DQN(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=dict(net_arch=[512, 512]),  # 神經網路架構
            learning_rate=1e-3,                       # 學習率
            buffer_size=100_000,                      # replay buffer 大小
            learning_starts=10_000,                   # 開始訓練前需要先收集多少步數的資料
            batch_size=256,                            # 每次更新時抽取的 batch 大小（調整為較合理的值）
            tau=0.01,                                 # soft update 的係數 (Polyak update)
            gamma=0.99,                               # 折扣因子
            target_update_interval=1_000,             # 每多少步數更新一次目標網路
            exploration_fraction=0.1,                 # 探索率從初始到最終下降需要經歷的 fraction
            exploration_initial_eps=1.0,              # 初始 epsilon
            exploration_final_eps=0.02,               # 最終 epsilon
            verbose=1,
            tensorboard_log="highway_dqn_tensorboard/",  # TensorBoard 紀錄路徑
            device=device,
        )

        # 建立驗證環境與回調
        eval_env = gym.make("intersection-v0")
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./logs_dqn/",
            log_path="./logs_dqn/",
            eval_freq=10_000,
            deterministic=True,
            render=False,
        )

        # 開始訓練
        model.learn(total_timesteps=int(1e6), callback=eval_callback)

        # 保存模型
        model.save(model_path)
        print("Model saved.")
    else:
        # 如果已經訓練過，直接載入模型
        model = DQN.load(model_path, device=device)

    # 測試模型
    # 在載入模型後，建立測試環境並設定 render_mode 為 "human" 以顯示畫面
    test_env = gym.make("intersection-v0", render_mode="human")

    for i in range(10):
        obs, info = test_env.reset()
        done, truncated = False, False
        episode_reward = 0.0
        while not (done or truncated):
            # 使用模型預測動作，並設定 deterministic=True 以確保行為確定性
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            episode_reward += reward
            test_env.render()

            # 為了避免畫面更新過快，可以加入短暫的延遲
            time.sleep(0.05)
        print(f"Test Episode {i+1} Reward: {episode_reward}")

    # 關閉環境
    test_env.close()

if __name__ == "__main__":
    main()
