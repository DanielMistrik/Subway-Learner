import keyboard
import torch
from stable_baselines3.common import policies
import numpy as np
import matplotlib.pyplot as plt

import env
import game
import view

ss_env = env.ImageSubwaySurferEnv()


class HumanPolicy(policies.BasePolicy):
    """
    A policy that queries the human for actions by showing the state
    """

    def __init__(self, pause_game=True, *args, **kwargs):
        super(HumanPolicy, self).__init__(ss_env.observation_space, ss_env.action_space, *args, **kwargs)
        self.pause_game = pause_game
        self.action = game.Action.NOOP
        self.observation_space = ss_env.observation_space
        self.action_space = ss_env.action_space

    def _predict(self, observation: np.ndarray, deterministic: bool = False):
        if self.pause_game:
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
        if self.pause_game:
            view.click_delayed_start()
        return torch.tensor(actions)

    def forward(self, *args, **kwargs):
        pass

    def _get_data(self) -> dict:
        return {}
