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
    find_play_button -
    :param accuracy:
    :return:
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


def detect_score(reader, x, y, w, h):
    """
    detect_score - Function that reads in the score of the game given the game window's dimensions and upper-left corner
     coordinates extracted from the detect_screen function (it has to be the one above, not a guess as this function
     works with a small highly-sensitive window). Also takes in an ocr reader of type easyocr.Reader. Is very noisy
    :return: The parsed score as an integer, if it's unable to parse it returns -1
    """
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    array_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(array_image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    result = reader.readtext(threshold)
    score = [detected_char[1] for detected_char in result]
    raw_score = ''.join(filter(str.isdigit, score))
    int_score = int(raw_score) if len(raw_score) > 0 else -1
    return int_score
