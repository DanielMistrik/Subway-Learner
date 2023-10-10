"""
view.py - Python script responsible for all CV
"""
import math

import cv2
import pyautogui
import numpy as np
import matplotlib.pyplot as plt


def find_play_button(accuracy=0.6):
    """
    find_play_button - Finds the green play button on the screen
    :param accuracy: The accuracy of the match, if you arent finding the play button
    successfully try lowering the accuracy
    :return: Coordinates of the center of the play button, if not found a tuple of Nones
    """
    template = cv2.imread('source_pictures//play_button_reduced.png', 0)
    w, h = template.shape[::-1]
    screenshot = pyautogui.screenshot()
    screen_array = np.array(screenshot)
    screen_gray = cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= accuracy)
    try:
        center_x = loc[::-1][0][0] + w // 2
        center_y = loc[::-1][1][0] + h // 2
        return center_x, center_y
    except:
        return None, None


def detect_cross():
    """
    Detects whether a cross appears on the screen. Is meant to remove any pop-ups on restart
    :return: Coordinates of cross if found, otherwise none
    """
    screenshot = pyautogui.screenshot()
    screen_array = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return _detect_color_median(screen_array, (60, 60, 182), (60, 60, 183), 20)


def detect_screen():
    """
    detect_screen - Will detect the subway surfer screen, its heuristics are any window
    whose width and height is above 100 pixels and its height is atleast 2 times the height.
    It assumes a unique window exists so it will return the first one it finds, otherwise None.
    Function only works if the background is a monotone and very light color
    :return: x,y,w,h where (x,y) is the upper-left coordinate and w,h are width/height in pixels
    """
    screenshot = pyautogui.screenshot()
    screen_array = np.array(screenshot)
    img = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    def func(value):
        return 0 if value > 240 else 255

    cleaned_image = np.vectorize(func)(np.mean(thresh, axis=-1)).astype(np.uint8)
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        if len(hull) == 4:
            x = np.array([hull[0][0], hull[1][0], hull[2][0], hull[3][0]])
            width, height = 0,0
            for i in range(1, 4):
                if math.isclose(x[0][0], x[i][0], rel_tol = 10):
                    height = abs(x[0][1]-x[i][1])
                    width = abs(x[0][0]-x[1 if i != 1 else 2][0])
                    break
            if width > 150 and height > 150 and 1.8 * width < height < 2.2 * width:
                sorted_x = x[np.argsort(x[:, 1])]
                i = int(sorted_x[0][0] > sorted_x[1][0])
                return sorted_x[i][0], sorted_x[i][1], width, height
    return None, None, None, None


def _detect_color_median(array_image, lower_bound, upper_bound, sensitivity=50):
    """
    _detect_color_median - Returns the median coordinates of a particular range of colours on the array image
    :param array_image: BGR converted cv2 standard numpy array,
    :param lower_bound: BGR lower bound for the colour range
    :param upper_bound: BGR upper bound for the colour range
    :param sensitivity: (Optional). Minimum number of coloured coordinates to be considered. Increase for regularization
    :return: Median coordinates of the pixels with a colour within the range
    """
    mask = cv2.inRange(array_image, lower_bound, upper_bound)
    coordinates = np.column_stack(np.where(mask > 0))
    median = np.median(coordinates, axis=0)[::-1]
    return (None, None) if len(coordinates) < sensitivity else (median[0], median[1])


def detect_player(x, y, w, h):
    """
    detect_player - Detects the subway surf character 'Jake' on the part of the
    screen given by x,y,w,h
    :param x: The x part of the upper-left coordinate of the screen to be considered
    :param y: The y part of the upper-left coordinate of the screen to be considered
    :param w: The width of the screen (in pixels) to be considered
    :param h: The height of the screen (in pixels) to be considered
    :return: A coordinate corresponding to some part of the player's head, if the player
    isn't found a tuple of Nones will be returned
    """
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    array_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return _detect_color_median(array_image, (220, 255, 255), (245, 255, 255))


def _detect_train_1(array_image):
    """
    Detects the center of the train with the door (it detects the door color)
    """
    return _detect_color_median(array_image, (90, 110, 2), (95, 113, 4))


def _detect_train_2(array_image):
    """
    Detects the white paint under the window
    """
    return _detect_color_median(array_image, (138, 138, 148), (141, 141, 154), 30)


def _detect_train_3(array_image):
    """
    Detects the under container blue shadow colour
    """
    return _detect_color_median(array_image, (89, 56, 15), (90, 57, 16))


def _detect_under_obstacle(array_image):
    """
    Detects the red-white obstacle which you can duck under, detect the wooden stand.
    """
    return _detect_color_median(array_image, (80, 95, 165), (90, 105, 170), 25)


def _detect_obstacle(array_image):
    """
    Detects any red-white obstacle, detects the red paint
    """
    return _detect_color_median(array_image, (10, 2, 216), (11, 3, 224))


def _detect_platform(array_image):
    """
    Detects a platform, detects the blue striped paint on the sides
    """
    return _detect_color_median(array_image, (78, 60, 62), (78, 60, 63), 200)

# Doesn't work at all, ignoring it for now <- Fix if needed
def _detect_wall(array_image):
    """
    Detects the wall/case, detects the brown rock colour
    """
    return _detect_color_median(array_image, (49, 57, 84), (57, 60, 90), 350)


def is_alive(x, y, w, h):
    """
    is_alive - Returns a boolean on whether the player is still in the game. Detects
    by colour matching to the golden star next to the score function. Assumes the coordinates
    and width/height point to this region.
    :return: True if player is alive, False otherwise
    """
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    array_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return _detect_color_median(array_image, (24, 116, 229), (38, 143, 251), 10)[0] is not None


def detect_labeled_obstacles(x, y, w, h):
    """
    detect_labeled_obstacles - Detects obstacles and returns a labeled coordinate+width+height of them in a dictionary
    :param y: The y part of the upper-left coordinate of the screen to be considered
    :param w: The width of the screen (in pixels) to be considered
    :param h: The height of the screen (in pixels) to be considered
    :return: Returns a dictionary of labelled obstacles in a vector containing the median coordinates of their location
    with the elements determining which part of the screen they were found. The structure is as follows:
    [left third, middle third, last third, left half, right half].
    """
    modified_y, modified_height = y+h/3, (2/3)*h
    # Capturing different parts of the screen
    array_left = cv2.cvtColor(np.array(pyautogui.screenshot(region=(x, modified_y, w*(3.5/9), modified_height))),
                              cv2.COLOR_RGB2BGR)
    array_middle = cv2.cvtColor(np.array(pyautogui.screenshot(region=(x+w*(3.5/9), modified_y, w*(2/9), modified_height))),
                                cv2.COLOR_RGB2BGR)
    array_right = cv2.cvtColor(np.array(pyautogui.screenshot(region=(x+w*(5.5/9), modified_y, w*(3.5/9), modified_height))),
                               cv2.COLOR_RGB2BGR)
    array_left_half = cv2.cvtColor(np.array(pyautogui.screenshot(region=(x, modified_y, w/2, modified_height))),
                               cv2.COLOR_RGB2BGR)
    array_right_half = cv2.cvtColor(np.array(pyautogui.screenshot(region=(x + w / 2, modified_y, w / 2, modified_height))),
        cv2.COLOR_RGB2BGR)

    # Collecting results for a function into a list
    def vectorize_func(f):
        return [f(array_left), f(array_middle), f(array_right), f(array_left_half), f(array_right_half)]

    # Collect actual data for each obstacle
    return_dict = {'train_1': vectorize_func(_detect_train_1), 'train_2': vectorize_func(_detect_train_2),
                   'train_3': vectorize_func(_detect_train_3), 'under_obstacle': vectorize_func(_detect_under_obstacle),
                   'obstacle': vectorize_func(_detect_obstacle), 'platform': vectorize_func(_detect_platform),
                   'wall': vectorize_func(_detect_wall)}

    return return_dict
