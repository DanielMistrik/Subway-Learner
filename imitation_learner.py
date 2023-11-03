import os
import time

from imitation.data import rollout

import view
import imitation
import numpy as np
import stable_baselines3.common.policies
from imitation.policies.base import SAC1024Policy
from imitation.policies.serialize import load_policy
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import NatureCNN

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
from imitation.algorithms.dagger import SimpleDAggerTrainer, ExponentialBetaSchedule

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
        view.click_pause()
        actions = []
        for obs in observation:
            action = 0
            def on_key_event(e):
                nonlocal action
                match e.name:
                    case 'up' | 'w':
                        action = 3
                        return False
                    case 'down' | 's':
                        action = 4
                        return False
                    case 'left' | 'a':
                        action = 1
                        return False
                    case 'right' | 'd':
                        action = 2
                        return False
                    case _:
                        action = 0
                        return False

            keyboard.hook(on_key_event)
            plt.imshow(obs[0])
            plt.axis('off')
            plt.ion()
            plt.show()
            plt.pause(0.25)
            keyboard.read_event()
            plt.close('all')
            plt.clf()
            plt.pause(0.25)
            actions.append([action])
        view.click_delayed_start()
        return torch.tensor(actions)

    def forward(self, *args, **kwargs):
        pass

    def _get_data(self) -> dict:
        return {}


class FeedForward64Policy(policies.ActorCriticCnnPolicy):
    """
    Beefed Up Policy to better understand the environment
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[256, 256, 128, 64])


expert = HumanPolicy()


def collect_rollouts(n):
    observations = np.empty((0, 1, 41, 45))
    next_observations = np.empty((0, 1, 41, 45))
    acts = np.array([])
    dones = np.array([])
    infos = np.array([])
    skeleton_transition = None
    prefix = 'rollout_data/'
    if os.path.exists(prefix + 'observations.npy'):
        observations = np.load(prefix+'observations.npy', allow_pickle=True)
        next_observations = np.load(prefix + 'next_observations.npy', allow_pickle=True)
        acts = np.load(prefix+'acts.npy', allow_pickle=True)
        dones = np.load(prefix+'dones.npy', allow_pickle=True)
        infos = np.load(prefix + 'infos.npy', allow_pickle=True)

    while observations.shape[0] < n or skeleton_transition is None:
        rollouts = rollout.rollout(
            expert,
            ss_env,
            rollout.make_sample_until(min_timesteps=5, min_episodes=None),
            unwrap=False,
            rng=rng,
        )
        temp_transition = rollout.flatten_trajectories(rollouts)
        skeleton_transition = temp_transition if skeleton_transition is None else skeleton_transition
        observations = np.concatenate((observations, temp_transition.obs), axis=0)
        next_observations = np.concatenate((next_observations, temp_transition.next_obs), axis=0)
        acts = np.concatenate((acts, temp_transition.acts), axis=0)
        infos = np.concatenate((infos, temp_transition.infos), axis=0)
        dones = np.concatenate((dones, temp_transition.dones), axis=0).astype(np.bool)
        # Save after every individual roll-out to prevent loss due to crash
        np.save(prefix + 'observations.npy', observations)
        np.save(prefix + 'next_observations.npy', next_observations)
        np.save(prefix + 'acts.npy', acts)
        np.save(prefix + 'infos.npy', infos)
        np.save(prefix + 'dones.npy', dones)
        # Print progress
        print(str(observations.shape[0])+"/"+str(n))

    state_parts = {'obs': observations, 'next_obs': next_observations, 'acts': acts, 'dones': dones, 'infos': infos}

    return imitation.data.types.Transitions(**state_parts)


def get_BC_trainer(pre_train=False):
    if pre_train:
        print("#####\nBeginning pre-training BC Learner\n#####")
        transitions = collect_rollouts(n=2000)
        bc_trainer = bc.BC(
            observation_space=ss_env.observation_space,
            action_space=ss_env.action_space,
            rng=rng,
            batch_size=4,
            demonstrations=transitions,
            policy=FeedForward64Policy(
                observation_space=ss_env.observation_space,
                action_space=ss_env.action_space,
                lr_schedule=lambda _: torch.finfo(torch.float32).max, )
        )
        bc_trainer.train(n_epochs=20)
        bc_trainer.policy.save('bc-learner')
        print("#####\nFinished pre-training BC Learner\n#####")
        return bc_trainer
    else:
        bc_trainer = bc.BC(
            observation_space=ss_env.observation_space,
            action_space=ss_env.action_space,
            rng=rng,
            batch_size=4,
            policy=FeedForward64Policy(
                observation_space=ss_env.observation_space,
                action_space=ss_env.action_space,
                lr_schedule=lambda _: torch.finfo(torch.float32).max, )
        )
        return bc_trainer


def train_DAgger(n=500, pre_train=False):
    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        bc_trainer = get_BC_trainer(pre_train)
        dagger_trainer = SimpleDAggerTrainer(
            venv=ss_env,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            scratch_dir=tmpdir,
            rng=rng,
            beta_schedule=ExponentialBetaSchedule(0.5),
        )
        dagger_trainer.train(n, rollout_round_min_timesteps=4, bc_train_kwargs={'n_epochs': 10})
    return dagger_trainer


def evaluate_learner(learner_policy, n=5):
    total_time_lasted = 0
    for episode in range(n):
        obs, done = ss_env.reset(), [False]
        tick = time.perf_counter()
        while not done[0]:
            if isinstance(obs, tuple):
                obs = obs[0]
            action, _ = learner_policy.predict(obs)
            obs, reward, done, info = ss_env.step(action)
        total_time_lasted += time.perf_counter() - tick
    print(str(total_time_lasted / n))


if __name__ == '__main__':
    #learner = train_DAgger(4000, pre_train=True)
    #learner.policy.save('imitation-learner')
    loaded_policy = stable_baselines3.common.policies.ActorCriticCnnPolicy.load('imitation-learner')
    evaluate_learner(loaded_policy)
    # pras_kay()
