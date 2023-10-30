"""
env.py - The openGym API class for subway surfer
"""
import time
from abc import ABC

import cv2
import gymnasium as gym, game, numpy as np
import pyautogui
from gym import spaces
from gymnasium.envs.registration import register

import view

NORMALIZING_CONSTANT = 100


def reward_function():
    return 1.0


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
        self.last_state = np.zeros((15,))
        self.count = 0

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
        reward = reward_function()
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


class ImageSubwaySurferEnv(gym.Env, ABC):
    subway_game: game.Game
    game_live: bool
    game_new: bool
    last_state: np.ndarray

    def __init__(self):
        self.action_space = spaces.Discrete(len(game.Action))
        self.observation_space = spaces.Box(low=0, high=12, shape=(41, 45, 1), dtype=np.uint8)
        self.num_envs = 1
        self.game_live = False
        self.game_new = True
        self.subway_game = game.Game()
        self.last_state = np.zeros((85, 85, 1))
        self.count = 0
        self.step_results = None

    def _get_feature_vector(self):
        x, y = self.subway_game.upper_right_screen_coordinates
        screen_array = pyautogui.screenshot(region=(x, y, self.subway_game.pixel_width_of_game_screen,
                                                    self.subway_game.pixel_height_of_game_screen))
        screenshot_array = np.array(screen_array)
        array_image = cv2.cvtColor(screenshot_array, cv2.COLOR_RGB2GRAY)
        array_image = array_image[280:-280, 80:-80]
        resizing_ratio = 6
        new_height = array_image.shape[0] // resizing_ratio
        new_width = array_image.shape[1] // resizing_ratio
        image_blocks = array_image[:new_height * resizing_ratio, :new_width * resizing_ratio].reshape(
            new_height, resizing_ratio, new_width, resizing_ratio)
        array_image = np.mean(image_blocks, axis=(1, 3)).reshape(new_height, new_width, 1)
        array_image = np.round(array_image * (12.0 / 255)).astype(np.uint8)
        is_alive = \
            view._detect_color_median(screenshot_array[50:90, 280:320], (229, 116, 24), (251, 143, 38), 10)[0] is not None
        return array_image, is_alive

    def step(self, action):
        # Execute one time step within the environment
        if self.game_new:
            self.game_live = True
            self.game_new = False
            self.subway_game.start()
        elif not self.game_live:
            self.subway_game.restart()
            self.game_live = True
        else:
            self.subway_game.action(action)
            if action != game.Action.NOOP:
                time.sleep(0.25)
            else:
                time.sleep(0.05)
        self.last_state = self._get_feature_vector()
        done = not self.last_state[1]
        reward = reward_function()
        next_state = self.last_state[0]
        return np.array([next_state]), np.array([reward]), np.array([done]), np.array([{"terminal_observation":next_state}])

    def reset(self, seed=None):
        # Reset the state of the environment to an initial state
        # Logic is that even if the game is running, in 3 seconds the player will hit an object, ending the game
        time.sleep(2)
        self.game_live = False
        self.count = 0
        return np.array([np.zeros((41, 45, 1), dtype=np.uint8)])

    # Is a function for a environment that can be asynchronized, here it makes no sense but
    # DAgger throws if it doesn't have it
    def step_async(self, action):
        view.click_delayed_start()
        self.step_results = self.step(action)

    def step_wait(self):
        view.click_pause()
        return self.step_results

register(
    id='SubwaySurferEnv-v1',
    entry_point='env:ImageSubwaySurferEnv',
)