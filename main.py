# -*- coding: utf-8 -*-
"""
Space Invaders AI
Autor: Twój Nick
Wersja: 2.0
Licencja: MIT
Opis: Profesjonalny, zaawansowany system AI do gry w Space Invaders z DQN, monitoringiem i testami.
"""
from core.screen_utils import find_game_region, grab_screen, save_lives_region
from ai.ai_utils import detect_player, detect_enemies, read_score, ai_decision, DQNAgent, detect_gameover, detect_menu, detect_lives, detect_menu_play_selected
from core.controller import perform_action
import time
import cv2
import numpy as np
import os
import pyautogui
from configs.config import TEMPLATE_PLAYER_PATH, TEMPLATE_ENEMY_PATH, SCORE_REGION
import torch
import pygetwindow as gw
import logging
from typing import Tuple, Optional
from game_ai import GameAI

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.FileHandler('log_ai.txt', 'a', 'utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def preprocess(img: np.ndarray) -> np.ndarray:
    """Przeskaluj obraz do rozmiaru 84x84."""
    img = cv2.resize(img, (84, 84))
    return img

def get_score_panel(img: np.ndarray) -> np.ndarray:
    """Wytnij region wyniku z obrazu."""
    y1, y2, x1, x2 = SCORE_REGION
    h, w = img.shape[:2]
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h))
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w))
    return img[y1:y2, x1:x2].copy()

def get_lives_panel(img: np.ndarray) -> np.ndarray:
    """Wytnij region z życiami."""
    return img[80:140, 40:180].copy()

def calc_reward(prev_score: int, score: int, done: bool, prev_enemies: list, enemies: list, prev_lives: int, lives: int) -> float:
    """Oblicz nagrodę dla agenta."""
    reward = 0.0
    if done:
        reward -= 2.0
    if score > prev_score:
        reward += 100.0 * (score - prev_score)  # BARDZO WYSOKA NAGRODA za SESZTRZELENIE
    if len(enemies) < len(prev_enemies):
        reward += 1.0 * (len(prev_enemies) - len(enemies))  # Niska nagroda za zniknięcie wroga
    if lives < prev_lives:
        reward -= 100.0 * (prev_lives - lives)  # BARDZO WYSOKA KARA
    return reward

def activate_dosbox_window() -> None:
    """Aktywuj okno DOSBox."""
    windows = gw.getWindowsWithTitle('DOSBox')
    if windows:
        win = windows[0]
        win.activate()
        time.sleep(0.05)

def main():
    game = GameAI()
    game.run()

if __name__ == '__main__':
    main() 