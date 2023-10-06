"""
- game.py -
This file is meant to serve as the API between the RL algorithm and the Subway Surfer Game. It should be able
to start, restart and control the game.
- Requirements -
The game is running, and it is on a monotonic white-ish background and is the only
window of its size and dimensions
"""
from enum import Enum
import pyautogui, time, numpy as np, cv2, matplotlib.pyplot as plt, view, os


ACTION_DURATION = 0.4


class Action(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class Game:
    upper_right_screen_coordinates: (int, int)
    center: (int, int)
    pixel_width_of_game_screen: int
    pixel_height_of_game_screen: int
    score_coordinates: (int, int)
    score_window_dimensions: (int, int)
    score: int
    additive_threshold: int

    def __init__(self):
        self.dimensions = pyautogui.size()
        x, y, w, h = view.detect_screen()
        self.pixel_width_of_game_screen = w
        self.pixel_height_of_game_screen = h
        self.center = (x + (w / 2), y + (h / 2))
        self.score_coordinates = (x + ((300 / 410) * w), y + ((50 / 765) * h))
        self.score_window_dimensions = ((100 / 410) * w, (35 / 765) * h)
        self.upper_right_screen_coordinates = x, y
        self.score = 0
        self.additive_threshold = 0

    def start(self) -> None:
        pyautogui.moveTo(self.center[0], self.center[1])
        pyautogui.click()
        time.sleep(2)

    def restart(self) -> None:
        # locateOnScreen function, use it
        play_button_center = view.find_play_button()
        while play_button_center[0] is None:
            play_button_center = view.find_play_button()
        time.sleep(1)
        pyautogui.moveTo(play_button_center[0], play_button_center[1])
        pyautogui.click()

    def action(self, action: Action) -> None:
        pyautogui.mouseDown()
        width = self.pixel_width_of_game_screen
        height = self.pixel_height_of_game_screen
        match action:
            case Action.LEFT:
                pyautogui.dragRel(-width / 3, 0, duration=ACTION_DURATION)
            case Action.RIGHT:
                pyautogui.dragRel(width / 3, 0, duration=ACTION_DURATION)
            case Action.UP:
                pyautogui.dragRel(0, -height / 3, duration=ACTION_DURATION)
            case Action.DOWN:
                pyautogui.dragRel(0, height / 3, duration=ACTION_DURATION)
            case Action.NOOP:
                return
        pyautogui.mouseUp()
        pyautogui.moveTo(self.center[0], self.center[1])

    def get_score(self, threshold: int = 500) -> int:
        self.additive_threshold += 1
        if self.additive_threshold >= threshold:
            self.score += 1
            self.additive_threshold = 0
        return self.score

    def is_player_alive(self) -> bool:
        x, y = self.score_coordinates
        w, h = self.score_window_dimensions
        x -= (60 / 410)*w
        w /= 2
        return view.is_alive(x, y, w, h)

    def get_player_location(self) -> (int, int):
        x, y = self.upper_right_screen_coordinates
        game_screen_third = (1/3)*self.pixel_height_of_game_screen
        y += game_screen_third
        x += 2
        w = self.pixel_width_of_game_screen - 5
        h = 2*game_screen_third
        results = view.detect_player(x, y, w, h)
        return (results[0]+2, results[1]+game_screen_third) if results[0] is not None else (None, None)

    def get_obstacles(self) -> dict:
        x, y = self.upper_right_screen_coordinates
        w, h = self.pixel_width_of_game_screen, self.pixel_height_of_game_screen
        return view.detect_labeled_obstacles(x, y, w, h)

    def get_discrete_player_location(self) -> [int, int]:
        player_x, player_y = self.get_player_location()
        return_vector = [0,0]
        if player_x is None or player_y is None:
            return [None, None]
        return_vector[0] = 0 if 165 <= player_x <= 235 else (-1 if player_x < 200 else 1)
        return_vector[1] = 0 if player_y > 470 else 1
        return return_vector

    def normalize_train_distance(self, coordinates):
        """
        normalize_train_distance - Normalizes the x,y coordinates and sets them up such that 1 is furthest away from
        the player and 0 is colliding with the player
        :param coordinates: The x,y coordinates of an object
        :return: Normalized coordinates (None is replaced by 2 as being not in sight)
        """
        if coordinates[0] is None:
            return 2, 2

        def norm(val):
            return 1-min(max((val-10)/70, 0), 1)

        return norm(coordinates[0]), norm(coordinates[1])

    def assign_y_dists(self, original_dist_matrix, index_list: [int], post_append_val: float = None,
                       pre_append_val: float = None) -> [float]:
        return_vec = []
        if pre_append_val is not None:
            return_vec.append(pre_append_val)
        for i in index_list:
            return_vec.append(original_dist_matrix[i][1])
        if post_append_val is not None:
            return_vec.append(post_append_val)
        return return_vec

    def assign_lanes(self, original_dist_matrix, lane) -> [float, float, float]:
        match lane:
            case -1:
                return self.assign_y_dists(original_dist_matrix, [3, 4], post_append_val=0)
            case 0:
                return self.assign_y_dists(original_dist_matrix, [0, 1, 2])
            case 1:
                return self.assign_y_dists(original_dist_matrix, [3, 4], pre_append_val=0)
        return [0, 0, 0]

    def get_train_locations(self, obstacle_dict: dict, lane: int) -> [float, float, float]:
        #  Train 1 -> 10 (Furthest we can observe), 80 (Impact) -> Normalize to [1,0] (Anything lower or greater set to min/max)
        #  Train 2 -> 10 (Furthest we can observe), 80 (Impact) -> Normalize to [1,0]
        #  Train 3 -> 10 (Furthest we can observe), 80 (Impact) -> Normalize to [1,0]
        train_1_lane_y_dists = self.assign_lanes(
            [self.normalize_train_distance(coord) for coord in obstacle_dict['train_1']], lane)
        train_2_lane_y_dists = self.assign_lanes(
            [self.normalize_train_distance(coord) for coord in obstacle_dict['train_2']], lane)
        train_3_lane_y_dists = self.assign_lanes(
            [self.normalize_train_distance(coord) for coord in obstacle_dict['train_3']], lane)
        trains_lane_y_dists = [train_1_lane_y_dists, train_2_lane_y_dists, train_3_lane_y_dists]
        lane_1_dist = sorted([arr[0] for arr in trains_lane_y_dists])[0]
        lane_2_dist = sorted([arr[1] for arr in trains_lane_y_dists])[0]
        lane_3_dist = sorted([arr[2] for arr in trains_lane_y_dists])[0]
        return [lane_1_dist, lane_2_dist, lane_3_dist]

    def get_under_obstacle_locations(self, obstacle_dict: dict, lane: int) -> [float, float, float]:
        return []

    def get_over_obstacle_locations(self, obstacle_dict: dict, lane: int) -> [float, float, float]:
        return []

    def get_wall_locations(self, obstacle_dict: dict, lane: int) -> [float, float, float]:
        return []

    def get_platform_locations(self, obstacle_dict: dict, lane: int) -> [float, float]:
        return []

    def get_state(self):
        """
        feature_vec =
        {
        player lane, player_jump, score,
        train lane 1, train lane 2, train lane 3,
        under_obstacle lane 1, under_obstacle lane 2, under_obstacle lane 3,
        over_obstacle lane 1, over_obstacle lane 2, over_obstacle lane 3,
        wall lane 1, wall lane 2, wall lane 3,
        platform lane 1, platform lane 3, is_alive
        }
        """
        feature_vec = []
        player_loc, score = self.get_discrete_player_location(), self.get_score()
        obstacle_dict, is_alive = self.get_obstacles(), self.is_player_alive()
        feature_vec += player_loc
        feature_vec.append(self.get_score())
        feature_vec += self.get_train_locations(obstacle_dict, player_loc[0])
        feature_vec += self.get_under_obstacle_locations(obstacle_dict, player_loc[0])
        feature_vec += self.get_over_obstacle_locations(obstacle_dict, player_loc[0])
        feature_vec += self.get_wall_locations(obstacle_dict, player_loc[0])
        feature_vec += self.get_platform_locations(obstacle_dict, player_loc[0])
        feature_vec.append(is_alive)
        return feature_vec


if __name__ == '__main__':
    test = Game()
    x, y = test.upper_right_screen_coordinates

    while(True):
        state = test.get_state()
        print(state)
        """
        coords = view.detect_labeled_obstacles(x, y, test.pixel_width_of_game_screen, test.pixel_height_of_game_screen)
        print(coords['train_1'])
        if coords[0] is not None:
            pyautogui.moveTo(coords[0]+x, coords[1]+y)
        """
    """
    test.start()
    test.action(Action.DOWN)
    test.action(Action.UP)
    test.action(Action.LEFT)
    test.action(Action.RIGHT)
    test.restart()
    """
