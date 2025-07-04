# -*- coding: utf-8 -*-
"""
Moduł screen_utils
Autor: Twój Nick
Wersja: 2.0
Licencja: MIT
Opis: Narzędzia do przechwytywania i lokalizacji regionu gry.
"""
import cv2
import numpy as np
import mss
from configs.config import MIN_GAME_WINDOW_SIZE
import pygetwindow as gw

def find_game_region():
    # Szukaj okna DOSBox po tytule
    windows = gw.getWindowsWithTitle('DOSBox')
    if windows:
        win = windows[0]
        x, y, w, h = win.left, win.top, win.width, win.height
        print(f'Znaleziono okno DOSBox: x={x}, y={y}, w={w}, h={h}')
        return {'top': y, 'left': x, 'width': w, 'height': h}
    else:
        print('Nie znaleziono okna DOSBox, używam cały monitor!')
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            return {'top': monitor['top'], 'left': monitor['left'], 'width': monitor['width'], 'height': monitor['height']}

def grab_screen(region):
    with mss.mss() as sct:
        img = np.array(sct.grab(region))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def save_lives_region(img):
    region = img[180:240, 0:180]  # Poszerzony w lewo do krawędzi okna
    cv2.imwrite('lives_debug.png', region)
    return region

def save_score_region(img):
    region = img[0:97, 0:185]  # Region punktacji wg @5.JPG
    cv2.imwrite('score_debug.png', region)
    return region 