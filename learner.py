import gym
import env

env = gym.make('SubwaySurferEnv-v0')

for episode in range(5):  # 5 episodes for demonstration
    observation = env.reset()
    done = False
    episode_reward = 0
    print("In episode ", episode)
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _, info = env.step(action)
        print("Taking ", action)
        episode_reward += reward

    print(f"Episode {episode + 1} Reward: {episode_reward}")

env.close()