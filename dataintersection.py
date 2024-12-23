import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt

def main():
    # 建立 intersection-v0，並且設定為灰階、stack_size=4
    env = gym.make("intersection-v0")
    env.unwrapped.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],
            "stack_size": 4,  # 堆疊 4 張灰階圖
            "observation_shape": (512, 512)
        },
        "action": {
            "type": "DiscreteMetaAction"
        },
        "simulation_frequency": 15,
        "policy_frequency": 5
    })

    obs, info = env.reset()
    # obs.shape => (4, 84, 84)
    for _ in range(10):
        obs, reward, done, trunc, info = env.step(env.action_space.sample())

    print("obs.shape =", obs.shape)  # 應該會顯示 (4, 84, 84)

    # 分別把 4 個 channel 顯示出來
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        # 取 obs[i]，shape => (84, 84)
        axes[i].imshow(obs[i], cmap='gray')
        axes[i].set_title(f"Channel {i}")
        axes[i].axis('off')

    plt.suptitle("Stacked Gray Frames (4×84×84) from intersection-v0")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
