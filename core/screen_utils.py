# -*- coding: utf-8 -*-
"""
Moduł screen_utils Space Invaders AI Pro
========================================

Narzędzia do przechwytywania i lokalizacji regionu gry.

Autor: lonter100
Wersja: 2.0
Licencja: MIT

Opis:
    Zawiera funkcje do przechwytywania obrazu ekranu,
    lokalizacji okna gry oraz zapisywania regionów debugowania.
"""
import cv2
import numpy as np
import mss
import pygetwindow as gw
import logging
from typing import Dict, Any, Optional
from configs.config import MIN_GAME_WINDOW_SIZE

# Konfiguracja loggera
logger = logging.getLogger(__name__)


def find_game_region() -> Dict[str, int]:
    """
    Znajdź region okna gry Space Invaders w DOSBox.
    
    Returns:
        Słownik z koordynatami regionu gry {'top', 'left', 'width', 'height'}
    """
    try:
        # Szukaj okna DOSBox po tytule
        windows = gw.getWindowsWithTitle('DOSBox')
        
        if windows:
            win = windows[0]
            x, y, w, h = win.left, win.top, win.width, win.height
            
            # Sprawdź minimalny rozmiar okna
            if w >= MIN_GAME_WINDOW_SIZE[0] and h >= MIN_GAME_WINDOW_SIZE[1]:
                logger.info(f'Znaleziono okno DOSBox: x={x}, y={y}, w={w}, h={h}')
                return {
                    'top': y, 
                    'left': x, 
                    'width': w, 
                    'height': h
                }
            else:
                logger.warning(f'Okno DOSBox jest za małe: {w}x{h}')
        else:
            logger.warning('Nie znaleziono okna DOSBox')
            
    except Exception as e:
        logger.error(f"Błąd podczas szukania okna DOSBox: {e}")
    
    # Fallback: użyj całego monitora
    logger.info('Używam całego monitora jako region gry')
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Główny monitor
            return {
                'top': monitor['top'], 
                'left': monitor['left'], 
                'width': monitor['width'], 
                'height': monitor['height']
            }
    except Exception as e:
        logger.error(f"Błąd podczas pobierania monitora: {e}")
        # Ostateczny fallback
        return {
            'top': 0, 
            'left': 0, 
            'width': 1920, 
            'height': 1080
        }


def grab_screen(region: Dict[str, int]) -> np.ndarray:
    """
    Przechwyć obraz z określonego regionu ekranu.
    
    Args:
        region: Słownik z koordynatami regionu {'top', 'left', 'width', 'height'}
        
    Returns:
        Obraz jako numpy array w formacie BGR
        
    Raises:
        Exception: W przypadku błędu przechwytywania ekranu
    """
    try:
        with mss.mss() as sct:
            img = np.array(sct.grab(region))
            # Konwertuj z BGRA do BGR
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        logger.error(f"Błąd podczas przechwytywania ekranu: {e}")
        raise


def save_lives_region(img: np.ndarray) -> np.ndarray:
    """
    Zapisz region z życiami gracza do pliku debugowania.
    
    Args:
        img: Obraz gry
        
    Returns:
        Region z życiami jako numpy array
    """
    try:
        # Region z życiami (poszerzony w lewo do krawędzi okna)
        region = img[180:240, 0:180]
        cv2.imwrite('lives_debug.png', region)
        logger.debug("Zapisano region żyć do lives_debug.png")
        return region
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania regionu żyć: {e}")
        return np.array([])


def save_score_region(img: np.ndarray) -> np.ndarray:
    """
    Zapisz region z wynikiem gry do pliku debugowania.
    
    Args:
        img: Obraz gry
        
    Returns:
        Region z wynikiem jako numpy array
    """
    try:
        # Region punktacji (wg analizy zrzutów ekranu)
        region = img[0:97, 0:185]
        cv2.imwrite('score_debug.png', region)
        logger.debug("Zapisano region wyniku do score_debug.png")
        return region
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania regionu wyniku: {e}")
        return np.array([])


def get_screen_resolution() -> tuple:
    """
    Pobierz rozdzielczość głównego monitora.
    
    Returns:
        Krotka (szerokość, wysokość) w pikselach
    """
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Główny monitor
            return (monitor['width'], monitor['height'])
    except Exception as e:
        logger.error(f"Błąd podczas pobierania rozdzielczości: {e}")
        return (1920, 1080)  # Domyślna rozdzielczość


def is_window_active(window_title: str) -> bool:
    """
    Sprawdź czy okno o podanym tytule jest aktywne.
    
    Args:
        window_title: Tytuł okna do sprawdzenia
        
    Returns:
        True jeśli okno jest aktywne, False w przeciwnym razie
    """
    try:
        windows = gw.getWindowsWithTitle(window_title)
        if windows:
            return windows[0].isActive
        return False
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania aktywności okna: {e}")
        return False 