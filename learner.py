import gymnasium as gym
import env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env


def try_random_learner():
    env = gym.make('SubwaySurferEnv-v0')

    for episode in range(5):
        observation, done, episode_reward = env.reset(), False, 0
        print("In episode ", episode)
        while not done:
            action = env.action_space.sample()
            observation, reward, done, _, info = env.step(action)
            if done:
                print("Done!")
            episode_reward += reward

        print(f"Episode {episode + 1} Reward: {episode_reward}")

    env.close()


def train_dqn_learner(steps=100) -> DQN:
    """
    Trains a DQN learner and saves it to dqn_subway
    :return:
    """
    ss_env = env.SubwaySurferEnv()
    model = DQN("MlpPolicy", ss_env).learn(total_timesteps=steps)
    model.save("dqn_subway")
    return model


def test_dqn_learner(learner: DQN, n=5):
    env = gym.make('SubwaySurferEnv-v0')
    total_reward = 0
    for episode in range(n):
        obs, done = env.reset(), False
        while not done:
            if isinstance(obs, tuple):
                obs = obs[0]
            action, _states = learner.predict(obs)
            obs, reward, done, _,  info = env.step(action)
            total_reward += reward
    print(str(total_reward/n))
    env.close()


if __name__ == '__main__':
    model = train_dqn_learner(10000)
    model = DQN.load("dqn_subway")
    test_dqn_learner(model)
    #try_random_learner()
    # ss_env = env.SubwaySurferEnv()
    # check_env(ss_env)