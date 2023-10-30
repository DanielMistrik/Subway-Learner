import time

import imitation
import numpy as np
import stable_baselines3.common.policies
from imitation.policies.serialize import load_policy
from stable_baselines3 import PPO

import env
import gymnasium as gym
import keyboard
import tempfile
import game
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import policies
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
import matplotlib.pyplot as plt
import torch
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

rng = np.random.default_rng(0)
ss_env = env.ImageSubwaySurferEnv()


class HumanPolicy(policies.BasePolicy):
    """
    A policy that queries the human for actions by showing the state
    """

    def __init__(self, *args, **kwargs):
        super(HumanPolicy, self).__init__(ss_env.observation_space, ss_env.action_space, *args, **kwargs)
        self.action = game.Action.NOOP
        self.observation_space = ss_env.observation_space
        self.action_space = ss_env.action_space

    def _predict(self, observation: np.ndarray, deterministic: bool = False):
        actions = []
        for obs in observation:
            action = 0

            def on_key_event(e):
                nonlocal action
                match e.name:
                    case 'up':
                        action = 3
                        return False
                    case 'down':
                        action = 4
                        return False
                    case 'left':
                        action = 1
                        return False
                    case 'right':
                        action = 2
                        return False
                    case _:
                        action = 0
                        return False

            keyboard.hook(on_key_event)
            plt.imshow(obs)
            plt.axis('off')
            plt.ion()
            plt.show()
            plt.pause(0.25)
            keyboard.read_event()
            plt.close('all')
            plt.clf()
            plt.pause(0.25)
            actions.append([action])
        return torch.tensor(actions)

    def forward(self, *args, **kwargs):
        pass

    def _get_data(self) -> dict:
        return {}


bc_trainer = bc.BC(
    observation_space=ss_env.observation_space,
    action_space=ss_env.action_space,
    rng=rng,
    batch_size=10,
)

expert = HumanPolicy()


def train_DAgger(n=500):
    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        dagger_trainer = SimpleDAggerTrainer(
            venv=ss_env,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            scratch_dir=tmpdir,
            rng=rng,
        )
        dagger_trainer.train(n, rollout_round_min_timesteps=32)
    return dagger_trainer


def evaluate_learner(dagger_policy, n=5):
    total_time_lasted = 0
    for episode in range(n):
        obs, done = ss_env.reset(), [False]
        tick = time.perf_counter()
        while not done[0]:
            if isinstance(obs, tuple):
                obs = obs[0]
            action, _ = dagger_policy.predict(obs)
            obs, reward, done, info = ss_env.step(action)
        total_time_lasted += time.perf_counter() - tick
    print(str(total_time_lasted / n))


if __name__ == '__main__':
    learner = train_DAgger(2_000)
    learner.policy.save('imitation-learner')
    loaded_policy = stable_baselines3.common.policies.ActorCriticPolicy.load('imitation-learner')
    evaluate_learner(loaded_policy)
    # pras_kay()
