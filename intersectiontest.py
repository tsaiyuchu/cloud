import gymnasium as gym
import highway_env
import time


def main():
    # 建立 intersection-v0 環境
    # 若想要在螢幕上顯示畫面，需要設定 render_mode="human"
    env = gym.make("intersection-v0", render_mode="human")

    # 重置環境，獲取初始觀測值
    obs, info = env.reset()

    done = False
    truncated = False

    # 隨機動作測試
    while not (done or truncated):
        # 隨機抽取動作
        action = env.action_space.sample()

        # 與環境互動
        obs, reward, done, truncated, info = env.step(action)

        # (選擇性) 顯示畫面
        env.render()

        # 適度延遲，避免畫面更新太快
        time.sleep(0.05)

    # 關閉環境
    env.close()


if __name__ == "__main__":
    main()
