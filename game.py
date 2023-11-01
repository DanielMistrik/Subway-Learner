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


def _assign_y_dists(original_dist_matrix, index_list: [int], post_append_val: float = None,
                    pre_append_val: float = None) -> [float]:
    return_vec = []
    if pre_append_val is not None:
        return_vec.append(pre_append_val)
    for i in index_list:
        return_vec.append(original_dist_matrix[i][1])
    if post_append_val is not None:
        return_vec.append(post_append_val)
    return return_vec


def _assign_lanes(original_dist_matrix, lane) -> [float, float, float]:
    match lane:
        case -1:
            return _assign_y_dists(original_dist_matrix, [3, 4], post_append_val=0)
        case 0:
            return _assign_y_dists(original_dist_matrix, [0, 1, 2])
        case 1:
            return _assign_y_dists(original_dist_matrix, [3, 4], pre_append_val=0)
    return [0, 0, 0]


def _normalize_train_distance(coordinates: [int, int], mean: int = 10, range: int = 70):
    """
    normalize_train_distance - Normalizes the x,y coordinates and sets them up such that 1 is furthest away from
    the player and 0 is colliding with the player
    :param coordinates: The x,y coordinates of an object
    :return: Normalized coordinates (None is replaced by 2 as being not in sight)
    """
    if coordinates[0] is None:
        return 2, 2

    def norm(val):
        return 1 - min(max((val - mean) / range, 0), 1)

    return norm(coordinates[0]), norm(coordinates[1])


def _get_train_locations(obstacle_dict: dict, lane: int) -> [float, float, float]:
    train_1_lane_y_dists = _assign_lanes(
        [_normalize_train_distance(coord) for coord in obstacle_dict['train_1']], lane)
    train_2_lane_y_dists = _assign_lanes(
        [_normalize_train_distance(coord) for coord in obstacle_dict['train_2']], lane)
    train_3_lane_y_dists = _assign_lanes(
        [_normalize_train_distance(coord) for coord in obstacle_dict['train_3']], lane)
    trains_lane_y_dists = [train_1_lane_y_dists, train_2_lane_y_dists, train_3_lane_y_dists]
    lane_1_dist = sorted([arr[0] for arr in trains_lane_y_dists])[0]
    lane_2_dist = sorted([arr[1] for arr in trains_lane_y_dists])[0]
    lane_3_dist = sorted([arr[2] for arr in trains_lane_y_dists])[0]
    return [lane_1_dist, lane_2_dist, lane_3_dist]


def _get_under_obstacle_locations(obstacle_dict: dict, lane: int) -> [float, float, float]:
    wood_barrier_y_dists = _assign_lanes(
        [_normalize_train_distance(coord, range=150) for coord in obstacle_dict['under_obstacle']], lane)
    red_barrier_y_dists = _assign_lanes(
        [_normalize_train_distance(coord, 5, 85) for coord in obstacle_dict['obstacle']], lane)

    def verify_obstacle(coords: (int, int)) -> int:
        if coords[0] is not None and coords[1] is not None and coords[0] <= 2 * coords[1]:
            return (coords[0] + coords[1]) / 2
        return 2

    return [verify_obstacle(coord_tuple) for coord_tuple in zip(wood_barrier_y_dists, red_barrier_y_dists)]


def _get_over_obstacle_locations(obstacle_dict: dict, lane: int) -> [float, float, float]:
    return _assign_lanes(
        [_normalize_train_distance(coord, 5, 85) for coord in obstacle_dict['obstacle']], lane)


# Broken <- Consult view.py for why
def _get_wall_locations(obstacle_dict: dict, lane: int) -> [float, float, float]:
    return _assign_lanes(
        [_normalize_train_distance(coord, 5, 65) for coord in obstacle_dict['wall']], lane)


def _get_platform_locations(obstacle_dict: dict, lane: int) -> [float, float]:
    platform_locations = _assign_lanes(
        [_normalize_train_distance(coord, 60, 250) for coord in obstacle_dict['platform']], lane)
    cleaned_plat_location = [x for x in platform_locations if 0 < x < 2]
    if len(cleaned_plat_location) > 0 and 0 < min(cleaned_plat_location) < 2:
        return [min(cleaned_plat_location), 2, min(cleaned_plat_location)]
    return [2, 2, 2]


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
        self.score = 1
        self.additive_threshold = 0

    def start(self) -> None:
        pyautogui.click(x=self.center[0], y=self.center[1])
        pyautogui.moveTo(x=self.center[0], y=self.center[1] - (330 / 765) * self.pixel_height_of_game_screen)
        time.sleep(2)

    def restart(self) -> None:
        play_button_center, detected_cross = view.find_play_button(), view.detect_cross()
        while detected_cross[0] is not None or play_button_center[0] is None:
            if detected_cross[0] is not None:
                pyautogui.click(x=detected_cross[0], y=detected_cross[1])
                time.sleep(1)
            play_button_center, detected_cross = view.find_play_button(), view.detect_cross()
        time.sleep(1)
        pyautogui.click(x=play_button_center[0], y=play_button_center[1])
        time.sleep(2)
        pyautogui.click(x=self.center[0], y=self.center[1] - (330 / 765) * self.pixel_height_of_game_screen)

    def action(self, action: Action) -> None:
        action = Action(action)
        match action:
            case Action.LEFT:
                pyautogui.press('left')
            case Action.RIGHT:
                pyautogui.press('right')
            case Action.UP:
                pyautogui.press('up')
            case Action.DOWN:
                pyautogui.press('down')
            case Action.NOOP:
                pass
            case _:
                print('not matched')

    def _is_player_alive(self) -> bool:
        x, y = self.score_coordinates
        w, h = self.score_window_dimensions
        x -= (60 / 410) * w
        w /= 2
        return view.is_alive(x, y, w, h)

    def _get_player_location(self) -> (int, int):
        x, y = self.upper_right_screen_coordinates
        game_screen_third = (1 / 3) * self.pixel_height_of_game_screen
        y += game_screen_third
        x += 2
        w = self.pixel_width_of_game_screen - 5
        h = 2 * game_screen_third
        results = view.detect_player(x, y, w, h)
        return (results[0] + 2, results[1] + game_screen_third) if results[0] is not None else (None, None)

    def _get_obstacles(self) -> dict:
        x, y = self.upper_right_screen_coordinates
        w, h = self.pixel_width_of_game_screen, self.pixel_height_of_game_screen
        return view.detect_labeled_obstacles(x, y, w, h)

    def _get_discrete_player_location(self) -> [int, int]:
        player_x, player_y = self._get_player_location()
        return_vector = [0, 0]
        if player_x is None or player_y is None:
            return [None, None]
        return_vector[0] = 0 if 165 <= player_x <= 235 else (-1 if player_x < 200 else 1)
        return_vector[1] = 0 if player_y > 470 else 1
        return return_vector

    def get_state(self):
        """
        feature_vec =
        {
        player lane, player_jump,
        train lane 1, train lane 2, train lane 3,
        under_obstacle lane 1, under_obstacle lane 2, under_obstacle lane 3,
        over_obstacle lane 1, over_obstacle lane 2, over_obstacle lane 3,
        platform lane 1, platform lane 2, platform lane 3, is_alive
        }
        """
        feature_vec = []
        player_loc = self._get_discrete_player_location()
        obstacle_dict, is_alive = self._get_obstacles(), self._is_player_alive()
        feature_vec += player_loc
        feature_vec += _get_train_locations(obstacle_dict, player_loc[0])
        feature_vec += _get_under_obstacle_locations(obstacle_dict, player_loc[0])
        feature_vec += _get_over_obstacle_locations(obstacle_dict, player_loc[0])
        feature_vec += _get_platform_locations(obstacle_dict, player_loc[0])
        feature_vec.append(is_alive)
        return feature_vec


if __name__ == '__main__':
    test = Game()
    x, y = test.upper_right_screen_coordinates
    screen_array = pyautogui.screenshot(region=(x, y, test.pixel_width_of_game_screen,
                                                test.pixel_height_of_game_screen))
    screenshot_array = np.array(screen_array)
    array_image = cv2.cvtColor(screenshot_array, cv2.COLOR_RGB2GRAY)
    array_image = array_image[280:-280, 80:-80]
    resizing_ratio = 7
    new_height = array_image.shape[0] // resizing_ratio
    new_width = array_image.shape[1] // resizing_ratio
    image_blocks = array_image[:new_height * resizing_ratio, :new_width * resizing_ratio].reshape(
        new_height, resizing_ratio, new_width, resizing_ratio)
    array_image = np.mean(image_blocks, axis=(1, 3))
    rescaled_image = np.round(array_image * (12.0 / 255)).astype(np.uint8)
    plt.imshow(rescaled_image)
    plt.axis('off')
    plt.show()
    print(view._detect_color_median(screenshot_array[:200, 200:], (229, 116, 24), (251, 143, 38), 10)[0] is not None)
