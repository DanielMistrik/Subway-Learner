"""
env.py - The openGym API class for subway surfer
"""
import time
from abc import ABC

import gym, game, numpy as np
from gym import spaces
from gym.envs.registration import register



def reward_function(feature_vector):
    return feature_vector[2]


def is_done(state):
    return state[-1] == 0


def get_median_of_index(array, index):
    return np.median(np.array([arr[index] for arr in array]))


def get_average_of_index(array, index, threshold=None):
    if threshold is None:
        return np.average(np.array([arr[index] for arr in array]))
    else:
        thresholded_result = [arr[index] for arr in array if arr[index] < threshold]
        return np.average(np.array(thresholded_result)) if len(thresholded_result) != 0 else threshold


class SubwaySurferEnv(gym.Env, ABC):
    subway_game: game.Game
    game_live: bool
    game_new: bool

    def __init__(self):
        self.action_space = spaces.Discrete(len(game.Action))
        self.observation_space = spaces.Box(low=-1, high=2, shape=(15,), dtype=np.float32)
        self.game_live = False
        self.game_new = True
        self.subway_game = game.Game()

    def _get_feature_vector(self):
        state_1 = self.subway_game.get_state()
        state_2 = self.subway_game.get_state()
        state_3 = self.subway_game.get_state()
        combined_states = [state_1, state_2, state_3]
        feature_vector = np.zeros((16,))
        # Average the 3, detect Nones (i.e. reduce Noise and return a numpy array). If player_loc[0] is None reject all but the score and alive
        # player lane, player jumping?, score <- Take average
        cleaned_combined_states = [x for x in combined_states if x[0] is not None]
        # Detect if we couldn't get player location for anything
        if len(cleaned_combined_states) == 0:
            feature_vector[2] = get_average_of_index(combined_states, 2)
        else:
            feature_vector[0] = get_median_of_index(cleaned_combined_states, 0)
            feature_vector[1] = get_median_of_index(cleaned_combined_states, 1)
            for i in range(2, 11):
                feature_vector[i] = get_average_of_index(cleaned_combined_states, i, 2)
            for i in range(11, 14):
                feature_vector[i] = min([cleaned_combined_states[j][i] for j in range(len(cleaned_combined_states))])
        feature_vector[-1] = get_median_of_index(combined_states, -1)
        return feature_vector.astype('float32')

    def step(self, action):
        # Execute one time step within the environment
        if self.game_new:
            self.game_live = True
            self.game_new = False
            self.subway_game.start()
        elif not self.game_live:
            self.subway_game.restart()
        else:
            self.subway_game.action(action)
        next_raw_state = self._get_feature_vector()
        reward, done = reward_function(next_raw_state), is_done(next_raw_state)
        next_state = next_raw_state[:-1]
        return next_state, reward, done, False, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        # Logic is that even if the game is running, in 4 seconds the player will hit an object, ending the game
        time.sleep(4)
        self.game_live = False
        return np.zeros((15,), dtype=np.float32), {}


register(
    id='SubwaySurferEnv-v0',
    entry_point='env:SubwaySurferEnv',
)

