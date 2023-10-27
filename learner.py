import time
import gymnasium as gym
import env
from stable_baselines3 import DQN
import os


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


def train_dqn_learner(learning_rate=1e-4, batch_size=128, gamma=0.99, steps=500) -> DQN:
    """
    Trains a DQN learner and saves it to dqn_subway
    :return:
    """
    ss_env = env.SubwaySurferEnv()
    model = DQN("MlpPolicy", ss_env, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                learning_starts=250).learn(total_timesteps=steps)
    model_name = "dqn_subway-learning_rate{}-batch_size{}-gamma{}-steps{}".format(learning_rate, batch_size, gamma, steps)
    model.save(model_name)
    return model


def test_dqn_learner(learner, n=5):
    subway_surf_env = gym.make('SubwaySurferEnv-v0')
    total_time_lasted = 0
    for episode in range(n):
        obs, done = subway_surf_env.reset(), False
        tick = time.perf_counter()
        while not done:
            if isinstance(obs, tuple):
                obs = obs[0]
            action, _ = learner(obs)
            obs, reward, done, _,  info = subway_surf_env.step(action)
        total_time_lasted += time.perf_counter() - tick
    print(str(total_time_lasted/n))
    subway_surf_env.close()


def grid_search(subway_surfer_env):
    possible_learning_rates = [5e-5, 8e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    possible_batch_sizes = [64, 128, 256]
    possible_gammas = [0.95, 0.97, 0.99, 0.995]

    for learning_rate in possible_learning_rates:
        for batch_size in possible_batch_sizes:
            for gamma in possible_gammas:
                print("Learning Rate:{}, Batch Size:{}. Gamma:{}".format(learning_rate, batch_size, gamma))
                model = train_dqn_learner(learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, steps=1000)
                dq_learner = lambda obs: model.predict(obs)
                test_dqn_learner(dq_learner)

if __name__ == '__main__':
    subway_surfer_env = gym.make('SubwaySurferEnv-v0')
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith('dqn_subway-learning_rate')]
    for file in files:
        print(file)
        model = DQN.load(file)
        dq_learner = lambda obs: model.predict(obs)
        test_dqn_learner(dq_learner, 20)
    # rand_learner = lambda obs: (subway_surfer_env.action_space.sample(), None)
    #try_random_learner()
    # ss_env = env.SubwaySurferEnv()
    # check_env(ss_env)