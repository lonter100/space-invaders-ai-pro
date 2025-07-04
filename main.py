# -*- coding: utf-8 -*-
"""
Space Invaders AI Pro
=====================

Zaawansowany system sztucznej inteligencji do gry Space Invaders,
wykorzystujący Deep Q-Learning (DQN) z monitoringiem i testami.

Autor: lonter100
Wersja: 2.0
Licencja: MIT

Opis:
    Profesjonalny system AI do automatycznej gry w Space Invaders.
    Wykorzystuje computer vision do wykrywania obiektów gry,
    Deep Q-Learning do podejmowania decyzji oraz
    zaawansowany system nagród i kar.
"""
import time
import cv2
import numpy as np
import os
import pyautogui
import torch
import pygetwindow as gw
import logging
from typing import Tuple, Optional

# Importy lokalnych modułów
from core.screen_utils import find_game_region, grab_screen, save_lives_region
from core.controller import perform_action
from configs.config import TEMPLATE_PLAYER_PATH, TEMPLATE_ENEMY_PATH, SCORE_REGION
from game_ai import GameAI

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('log_ai.txt', 'a', 'utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Przeskaluj obraz do rozmiaru 84x84 dla sieci neuronowej.
    
    Args:
        img: Obraz wejściowy jako numpy array
        
    Returns:
        Przeskalowany obraz o rozmiarze 84x84
    """
    img = cv2.resize(img, (84, 84))
    return img


def get_score_panel(img: np.ndarray) -> np.ndarray:
    """
    Wytnij region wyniku z obrazu gry.
    
    Args:
        img: Obraz gry
        
    Returns:
        Region z wynikiem gry
    """
    y1, y2, x1, x2 = SCORE_REGION
    h, w = img.shape[:2]
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h))
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w))
    return img[y1:y2, x1:x2].copy()


def get_lives_panel(img: np.ndarray) -> np.ndarray:
    """
    Wytnij region z życiami gracza.
    
    Args:
        img: Obraz gry
        
    Returns:
        Region z wyświetlanymi życiami
    """
    return img[80:140, 40:180].copy()


def calc_reward(prev_score: int, score: int, done: bool, 
                prev_enemies: list, enemies: list, 
                prev_lives: int, lives: int) -> float:
    """
    Oblicz nagrodę dla agenta na podstawie stanu gry.
    
    Args:
        prev_score: Poprzedni wynik
        score: Aktualny wynik
        done: Czy gra się zakończyła
        prev_enemies: Poprzednia lista wrogów
        enemies: Aktualna lista wrogów
        prev_lives: Poprzednia liczba żyć
        lives: Aktualna liczba żyć
        
    Returns:
        Wartość nagrody (float)
    """
    reward = 0.0
    
    if done:
        reward -= 2.0  # Kara za przegraną
        
    if score > prev_score:
        reward += 100.0 * (score - prev_score)  # Wysoka nagroda za punkty
        
    if len(enemies) < len(prev_enemies):
        reward += 1.0 * (len(prev_enemies) - len(enemies))  # Nagroda za zniszczenie wroga
        
    if lives < prev_lives:
        reward -= 100.0 * (prev_lives - lives)  # Wysoka kara za utratę życia
        
    return reward


def activate_dosbox_window() -> None:
    """
    Aktywuj okno DOSBox, aby AI mogło z nim współpracować.
    """
    windows = gw.getWindowsWithTitle('DOSBox')
    if windows:
        win = windows[0]
        win.activate()
        time.sleep(0.05)
        logger.info("Aktywowano okno DOSBox")
    else:
        logger.warning("Nie znaleziono okna DOSBox")


def main():
    """
    Główna funkcja uruchamiająca system AI.
    """
    try:
        logger.info("Uruchamianie Space Invaders AI Pro...")
        game = GameAI()
        game.run()
    except KeyboardInterrupt:
        logger.info("Przerwano przez użytkownika")
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania: {e}")
        raise


if __name__ == '__main__':
    main() 