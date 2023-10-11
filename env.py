"""
env.py - The openGym API class for subway surfer
"""
import time
from abc import ABC

import gymnasium as gym, game, numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register


def reward_function(feature_vector, action):
    # Apply basic rules (if violated a negative reward is given)
    game_action = game.Action(action)
    # If the player is on a corner-lane they cannot try to leave the game
    if feature_vector[0] == -1 and game_action == game.Action.LEFT or \
            feature_vector[0] == 1 and game_action == game.Action.RIGHT:
        return -1000
    # If the player is in front of a train, they cannot duck or jumo
    if 0 < feature_vector[int(3 + feature_vector[0])] < 0.7 and game_action in \
            [game.Action.DOWN, game.Action.NOOP]:
        return -5
    # Do not run into a train by changing lane
    direction = 1 if game_action == game.Action.RIGHT else (-1 if game_action == game.Action.LEFT else 0)
    if feature_vector[3 + direction] < 1:
        return -2

    return 1


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
    last_state: np.ndarray

    def __init__(self):
        self.action_space = spaces.Discrete(len(game.Action))
        self.observation_space = spaces.Box(low=-1, high=2, shape=(14,), dtype=np.float32)
        self.game_live = False
        self.game_new = True
        self.subway_game = game.Game()
        self.last_state = np.zeros((15, ))

    def _get_feature_vector(self):
        state_1 = self.subway_game.get_state()
        state_2 = self.subway_game.get_state()
        state_3 = self.subway_game.get_state()
        combined_states = [state_1, state_2, state_3]
        feature_vector = np.zeros((15,))
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
        # Get the reward on the current state given the action
        reward = reward_function(self.last_state, action)
        if self.game_new:
            self.game_live = True
            self.game_new = False
            self.subway_game.start()
        elif not self.game_live:
            self.subway_game.restart()
            self.game_live = True
        else:
            self.subway_game.action(action)
        self.last_state = self._get_feature_vector()
        done = bool(is_done(self.last_state))
        next_state = self.last_state[:-1]
        return next_state, reward, done, False, {}

    def reset(self, seed=None):
        # Reset the state of the environment to an initial state
        # Logic is that even if the game is running, in 3 seconds the player will hit an object, ending the game
        time.sleep(3)
        self.game_live = False
        return np.zeros((14,), dtype=np.float32), {}


register(
    id='SubwaySurferEnv-v0',
    entry_point='env:SubwaySurferEnv',
)
