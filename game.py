"""
- game.py -
This file is meant to serve as the API between the RL algorithm and the Subway Surfer Game. It should be able
to start, restart and control the game.
- Requirements -
The game is running, and it is on a monotonic white-ish background and is the only
window of its size and dimensions
"""
from enum import Enum

import pyautogui, time, numpy as np, cv2, matplotlib.pyplot as plt
import view
import easyocr
import os

ACTION_DURATION = 0.4


class Action(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class Game:
    dimensions: (int, int)
    center: (int, int)
    pixel_width_of_game_screen: int
    pixel_height_of_game_screen: int
    score_coordinates: (int, int)
    score_window_dimensions: (int, int)
    reader: easyocr.Reader

    def __init__(self):
        self.dimensions = pyautogui.size()
        x, y, w, h = view.detect_screen()
        self.pixel_width_of_game_screen = w
        self.pixel_height_of_game_screen = h
        self.center = (x + (w / 2), y + (h / 2))
        self.score_coordinates = (x + ((300 / 410) * w), y + ((50 / 765) * h))
        self.score_window_dimensions = ((100 / 410) * w, (35 / 765) * h)
        self.reader = easyocr.Reader(['en'])

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

    def get_score(self) -> int:
        x, y = self.score_coordinates
        w, h = self.score_window_dimensions
        return view.detect_score(self.reader, x, y, w, h)


if __name__ == '__main__':
    test = Game()
    while(True):
        print(test.get_score())
    """
    test.start()
    test.action(Action.DOWN)
    test.action(Action.UP)
    test.action(Action.LEFT)
    test.action(Action.RIGHT)
    test.restart()
    """
