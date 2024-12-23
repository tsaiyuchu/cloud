import random
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import gymnasium as gym
import highway_env
import os
from tqdm import tqdm

# -------------------
# 一些超參數
# -------------------
obs_shape = (4, 84, 84)  # (channel=4, height=84, width=84)
hidden_dim = 512
lr = 1e-4
gamma = 0.997
grad_norm = 40

# R2D2 相關參數
burn_in_steps = 40
learning_steps = 40
forward_steps = 5
seq_len = burn_in_steps + learning_steps + forward_steps  # 85
batch_size = 8         # 每次訓練取多少條序列
buffer_capacity = 200  # 簡化示範，用小一點就好 (真實應該大一些)
training_steps = 100000
num_actors = 2         # 示範用小一點

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------
# 建立 highway_env 的 intersection 環境
# -------------------
def create_env():
    env = gym.make("intersection-v0")
    env.unwrapped.configure({
        "observation": {
            "type": "GrayscaleObservation",  # Convert to grayscale
            "weights": [0.2989, 0.5870, 0.1140],
            "stack_size": 4,  # Stack 4 consecutive frames
            "observation_shape": (84, 84)
        },
        "action": {
            "type": "DiscreteMetaAction"
        },
        "simulation_frequency": 15,
        "policy_frequency": 5
    })
    return env


# -------------------
# R2D2 網路
# -------------------
class R2D2Network(nn.Module):
    def __init__(self, action_dim, hidden_dim=hidden_dim):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # CNN 特徵萃取
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),  # Input is 4 stacked frames
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Flatten(),                 # => shape: (batch_size, 3136)
            nn.Linear(3136, 512),
            nn.ReLU(True),
        )

        # 將 feature + one-hot(action) + reward => LSTM
        self.lstm = nn.LSTM(512 + action_dim + 1, hidden_dim, batch_first=True)

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

    def forward_sequence(self, obs_seq, action_seq, reward_seq, init_hidden=None):
        """
        用於一次性處理「batch_size 條序列」的 forward。
        形狀:
          obs_seq:    (batch_size, seq_len, 4, 84, 84)
          action_seq: (batch_size, seq_len)       # 每一步是一個 int action
          reward_seq: (batch_size, seq_len)       # 每一步是一個 float reward
        回傳:
          q_seq:      (batch_size, seq_len, action_dim)
          hidden:     (h, c)  # LSTM 最後的隱狀態
        """
        b, s, c, h, w = obs_seq.shape
        # 1) CNN feature: 先 reshape => (b*s, 4, 84, 84)
        obs_seq_2d = obs_seq.view(b*s, c, h, w)          # (b*s, 4, 84, 84)
        feat_2d = self.feature(obs_seq_2d)               # (b*s, 512)
        feat_seq = feat_2d.view(b, s, -1)                # (b, s, 512)

        # 2) 把 action 做 one-hot
        #    action_seq => shape: (b, s) 裡面是動作 id
        action_onehot = nn.functional.one_hot(action_seq.long(), self.action_dim).float()
        # => (b, s, action_dim)

        # 3) 把 reward reshape => (b, s, 1)
        reward_seq = reward_seq.unsqueeze(-1)

        # 4) 拼接成 LSTM input => (b, s, 512 + action_dim + 1)
        lstm_input = torch.cat([feat_seq, action_onehot, reward_seq], dim=-1)

        # 5) 丟進 LSTM
        lstm_out, hidden = self.lstm(lstm_input, init_hidden)  # (b, s, hidden_dim), (h, c)

        # 6) 用 advantage, value 得到 Q
        adv = self.advantage(lstm_out)  # (b, s, action_dim)
        val = self.value(lstm_out)      # (b, s, 1)
        q_seq = val + adv  # broadcast => shape: (b, s, action_dim)

        return q_seq, hidden


# -------------------
# 儲存「整條 episode」的資料，用於後續切分序列
# -------------------
class EpisodeBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def store(self, obs, action, reward, done):
        """
        obs: shape (4, 84, 84)  (因為 highway_env 已經把4張堆疊在同一個維度)
        action: int
        reward: float
        done: bool
        """
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.obs)


# -------------------
# 多條 episode 的 Replay Buffer，用來隨機抽取序列
# -------------------
class SequenceReplayBuffer:
    def __init__(self, capacity, seq_len):
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes = []  # 存 EpisodeBuffer
        self.size_eps = 0   # 真實的 episode 數量

    def store_episode(self, ep_buffer: EpisodeBuffer):
        """將一整條 episode 存進 buffer"""
        if self.size_eps >= self.capacity:
            # 簡單示範: 滿了就丟最舊
            self.episodes.pop(0)
        else:
            self.size_eps += 1

        self.episodes.append(ep_buffer)

    def sample_sequences(self, batch_size):
        """
        隨機從所有 episode 中，抽 batch_size 條「長度 = seq_len」的序列。
        如果某 episode 長度 < seq_len，就跳過。
        這裡的實作是簡易示範，真實場景可加入優先級經驗回放等。
        回傳:
          obs_batch:    (batch_size, seq_len, 4, 84, 84)
          action_batch: (batch_size, seq_len)
          reward_batch: (batch_size, seq_len)
          done_batch:   (batch_size, seq_len)
        """
        valid_eps = [ep for ep in self.episodes if len(ep) >= self.seq_len]
        if len(valid_eps) == 0:
            return None

        sampled_eps = random.choices(valid_eps, k=batch_size)

        obs_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []

        for ep in sampled_eps:
            start_idx = random.randint(0, len(ep) - self.seq_len)
            end_idx = start_idx + self.seq_len

            obs_seq = ep.obs[start_idx:end_idx]         # list of [4, 84, 84]
            action_seq = ep.actions[start_idx:end_idx]  # list of int
            reward_seq = ep.rewards[start_idx:end_idx]  # list of float
            done_seq = ep.dones[start_idx:end_idx]      # list of bool

            obs_batch.append(np.array(obs_seq, dtype=np.float32))
            action_batch.append(np.array(action_seq, dtype=np.int64))
            reward_batch.append(np.array(reward_seq, dtype=np.float32))
            done_batch.append(np.array(done_seq, dtype=bool))

        # 轉成 tensor
        obs_batch_t = torch.from_numpy(np.stack(obs_batch, axis=0)).to(device)
        action_batch_t = torch.from_numpy(np.stack(action_batch, axis=0)).to(device)
        reward_batch_t = torch.from_numpy(np.stack(reward_batch, axis=0)).to(device)
        done_batch_t = torch.from_numpy(np.stack(done_batch, axis=0)).to(device)

        return obs_batch_t, action_batch_t, reward_batch_t, done_batch_t

    def size(self):
        return self.size_eps


# -------------------
# Actor：與環境互動，收集整條 episode 並放入 Buffer
# -------------------
class Actor(mp.Process):
    def __init__(self, epsilon, global_model, buffer, worker_id=0):
        super().__init__()
        self.epsilon = epsilon
        # 為了讓 Actor 也能 forward，需要一份 model
        self.local_model = R2D2Network(global_model.action_dim, hidden_dim).to(device)
        self.local_model.load_state_dict(global_model.state_dict())

        self.buffer = buffer
        self.env = create_env()
        self.worker_id = worker_id

    def run(self):
        obs, _ = self.env.reset()
        done = False

        ep_buffer = EpisodeBuffer()

        # 取得第一個 obs (env 會回傳 (obs, info)，其中 obs shape=[(4,84,84)])
        while not done:
            # Epsilon-greedy
            if random.random() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                # 使用 local_model 做單步推理
                # 這裡是「單一步 (batch_size=1, seq_len=1)」的 forward
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)

                # 動作、獎勵先設 0 =>  (1,1) 全 0
                dummy_action = torch.zeros((1,1), dtype=torch.long).to(device)
                dummy_reward = torch.zeros((1,1), dtype=torch.float32).to(device)

                q_seq, _ = self.local_model.forward_sequence(obs_t, dummy_action, dummy_reward)
                # q_seq shape: (1,1, action_dim)
                action = q_seq.squeeze(0).squeeze(0).argmax().item()

            next_obs, reward, done, _, _ = self.env.step(action)
            # next_obs shape: [(4,84,84)], reward: float, done: bool

            # 存到 episode buffer
            ep_buffer.store(obs[0], action, reward, done)

            obs = next_obs

        # episode 結束，放入 replay buffer
        self.buffer.store_episode(ep_buffer)
        print(f"[Actor-{self.worker_id}] Episode done, length={len(ep_buffer)}.")


# -------------------
# Learner：對序列資料做訓練
# -------------------
class Learner:
    def __init__(self, global_model, buffer):
        self.global_model = global_model.to(device)
        self.buffer = buffer
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=lr)

    def train(self):
        for step in tqdm(range(training_steps), desc="Training Progress"):
            # 如果資料不夠，就跳過
            if self.buffer.size() < 1:
                continue

            batch = self.buffer.sample_sequences(batch_size)
            if batch is None:
                continue

            obs_batch, action_batch, reward_batch, done_batch = batch
            # obs_batch: (b, seq_len, 4, 84, 84)
            # action_batch: (b, seq_len)
            # reward_batch: (b, seq_len)
            # done_batch: (b, seq_len)

            # 前 burn_in_steps 不計算 loss，用來 warm-up hidden state
            burn_in_obs = obs_batch[:, :burn_in_steps]
            burn_in_act = action_batch[:, :burn_in_steps]
            burn_in_rew = reward_batch[:, :burn_in_steps]

            learn_obs = obs_batch[:, burn_in_steps:]
            learn_act = action_batch[:, burn_in_steps:]
            learn_rew = reward_batch[:, burn_in_steps:]
            learn_done = done_batch[:, burn_in_steps:]

            # ---- 1) 先做 burn-in，取得隱狀態 ----
            with torch.no_grad():
                _, hidden = self.global_model.forward_sequence(
                    burn_in_obs, burn_in_act, burn_in_rew, init_hidden=None
                )

            # ---- 2) 用後面的 (learning_steps + forward_steps) 做訓練 ----
            # 先一次性 forward 全序列
            q_seq, _ = self.global_model.forward_sequence(
                learn_obs, learn_act, learn_rew, init_hidden=hidden
            )  # (b, seq_len - burn_in_steps, action_dim)

            # 我們只計算前 learning_steps 的 TD loss
            # 但也要用到 forward steps 來算 n-step return
            # 這裡僅示範 1-step Q-learning，可自行改成 n-step。
            # 先把我們關心的部分切出 (前面 learning_steps)
            q_learn = q_seq[:, :learning_steps]     # (b, learning_steps, action_dim)

            # 取得對應的 action & reward & done
            act_learn = learn_act[:, :learning_steps]      # (b, learning_steps)
            rew_learn = learn_rew[:, :learning_steps]      # (b, learning_steps)
            done_learn = learn_done[:, :learning_steps]    # (b, learning_steps)

            # 取得下一步的 Q
            # 這裡是 single-step Q-learning 做示範
            # next Q = q_seq[:, 1:(learning_steps+1)]
            q_next = q_seq[:, 1:(learning_steps+1)]  # shape: (b, learning_steps, action_dim)

            # target = r + gamma * max Q_next
            # done 時 Q_next=0
            q_next_max = q_next.max(dim=-1)[0]               # (b, learning_steps)
            targets = rew_learn + gamma * q_next_max * (1 - done_learn.float())

            # gather 選出對應動作的 Q
            pred_q = q_learn.gather(dim=-1, index=act_learn.unsqueeze(-1)).squeeze(-1)  # (b, learning_steps)

            loss = nn.functional.mse_loss(pred_q, targets)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.global_model.parameters(), grad_norm)
            self.optimizer.step()

            if (step+1) % 100 == 0:
                print(f"[Learner] step={step+1}, loss={loss.item():.4f}")

        print("Training Done!")


# -------------------
# Main
# -------------------
def main():
    # 建立全域共享的 model
    env_tmp = create_env()
    action_dim = env_tmp.action_space.n
    global_model = R2D2Network(action_dim).to(device)

    # 建立序列型 replay buffer
    buffer = SequenceReplayBuffer(capacity=buffer_capacity, seq_len=seq_len)

    # 這裡示範單機多進程收集資料 => 也可以改成單進程
    actors = []
    for i in range(num_actors):
        actor = Actor(epsilon=0.2, global_model=global_model, buffer=buffer, worker_id=i)
        actors.append(actor)

    # 啟動每個 actor
    for actor in actors:
        actor.start()

    # 建立 Learner
    learner = Learner(global_model, buffer)

    # 在這裡做簡單的「先等所有 Actor 結束後再學習」的流程
    # 你也可以改成「Actor 不斷跑，Learner 不斷 train」的並行架構 (需用 mp.Queue / Pipe 等)
    for actor in actors:
        actor.join()

    # 開始訓練
    learner.train()

    # 訓練完儲存 model
    torch.save(global_model.state_dict(), "r2d2_model.pt")
    print("Model saved!")

    # 簡單測試
    test_env = create_env()
    obs, _ = test_env.reset()
    done = False
    total_reward = 0
    # LSTM hidden state 置 0
    hidden = None

    while not done:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)

        dummy_action = torch.zeros((1,1), dtype=torch.long).to(device)
        dummy_reward = torch.zeros((1,1), dtype=torch.float32).to(device)

        with torch.no_grad():
            q_seq, hidden = global_model.forward_sequence(obs_t, dummy_action, dummy_reward, hidden)
            action = q_seq.squeeze(0).squeeze(0).argmax().item()

        obs, reward, done, _, _ = test_env.step(action)
        total_reward += reward
        test_env.render()

    print(f"Test Episode Done, total_reward={total_reward}")


if __name__ == "__main__":
    main()
