import numpy as np
import env
import gymnasium as gym
import keyboard
import game
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import policies
from gymnasium.envs.registration import register
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
            plt.pause(0.1)
            keyboard.read_event()
            plt.close()
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
    batch_size=8,
)

expert = HumanPolicy()


def train_DAgger():
    dagger_trainer = SimpleDAggerTrainer(
        venv=ss_env,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        scratch_dir='./scratch',
        rng=rng,
    )
    dagger_trainer.train(10, rollout_round_min_episodes=1, rollout_round_min_timesteps=1)
    return dagger_trainer


def evaluate_learner(dagger_learner):
    reward, _ = evaluate_policy(dagger_learner.policy, ss_env, 10)
    print("Reward:", reward)


if __name__ == '__main__':
    learner = train_DAgger()
    evaluate_learner(learner)
    # pras_kay()
