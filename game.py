"""
- game.py -
This file is meant to serve as the API between the RL algorithm and the Subway Surfer Game. It should be able
to start, restart and control the game.
- Requirements -
The game is running with the dimensions specified below and is in the center of the screen
"""
from enum import Enum

import pyautogui, time

PIXEL_WIDTH_OF_GAME_SCREEN = 200
PIXEL_HEIGHT_OF_GAME_SCREEN = 550
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

    def __init__(self):
        self.dimensions = pyautogui.size()
        self.center = (self.dimensions[0]/2, self.dimensions[1]/2)

    def start(self) -> None:
        pyautogui.moveTo(self.center[0], self.center[1])
        time.sleep(2)
        pyautogui.click()

    def restart(self) -> None:
        pyautogui.click(x=self.center[0] + PIXEL_WIDTH_OF_GAME_SCREEN/2 - 5,
                        y=self.center[1] + PIXEL_HEIGHT_OF_GAME_SCREEN/2 + 5)
        time.sleep(0.25)
        pyautogui.click(x=self.center[0] + PIXEL_WIDTH_OF_GAME_SCREEN / 2 - 5,
                        y=self.center[1] + PIXEL_HEIGHT_OF_GAME_SCREEN / 2 + 5)

    def action(self, action: Action) -> None:
        pyautogui.mouseDown()
        match action:
            case Action.LEFT:
                pyautogui.dragRel(-PIXEL_WIDTH_OF_GAME_SCREEN/3, 0, duration=ACTION_DURATION)
            case Action.RIGHT:
                pyautogui.dragRel(PIXEL_WIDTH_OF_GAME_SCREEN/3, 0, duration=ACTION_DURATION)
            case Action.UP:
                pyautogui.dragRel(0, PIXEL_HEIGHT_OF_GAME_SCREEN/3, duration=ACTION_DURATION)
            case Action.DOWN:
                pyautogui.dragRel(0, -PIXEL_HEIGHT_OF_GAME_SCREEN / 3, duration=ACTION_DURATION)
            case Action.NOOP:
                return
        pyautogui.mouseUp()
        pyautogui.moveTo(self.center[0], self.center[1])


if __name__ == '__main__':
    time.sleep(1)
    test = Game()
    test.start()
    test.action(Action.LEFT)
    test.action(Action.RIGHT)
    test.action(Action.UP)
    test.action(Action.DOWN)
    test.action(Action.NOOP)
    test.action(Action.LEFT)
    test.restart()