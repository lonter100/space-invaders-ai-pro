# -*- coding: utf-8 -*-
"""
Moduł sterowania Space Invaders AI Pro
======================================

Obsługa sterowania ruchem i strzałem w grze Space Invaders.

Autor: lonter100
Wersja: 2.0
Licencja: MIT

Opis:
    Zawiera funkcje do symulacji naciśnięć klawiszy i myszki
    w celu sterowania grą Space Invaders uruchomioną w DOSBox.
    Obsługuje zarówno akcje gry jak i nawigację w menu.
"""
import pyautogui
import time
import keyboard
import subprocess
import os
import logging
from typing import Optional
from configs.config import ACTION_DELAY, MENU_DELAY

# Konfiguracja loggera
logger = logging.getLogger(__name__)

# =============================================================================
# STAŁE STEROWANIA
# =============================================================================

# Akcje ruchu
MOVE_LEFT = 'left'
MOVE_RIGHT = 'right'
MOVE_UP = 'up'
MOVE_DOWN = 'down'
SHOOT = 'space'

# Akcje menu
MENU_SELECT = 'menu_select'
MENU_UP = 'menu_up'
MENU_DOWN = 'menu_down'

# Ścieżka do AutoHotkey.exe (v2)
AHK_PATH = r"C:\Program Files\AutoHotkey\v2\AutoHotkey.exe"
AHK_SCRIPT = os.path.abspath("shoot.ahk")

# Mapowanie nazw klawiszy na kody Windows/ASCII
KEYCODES = {
    'a': ord('a'),      # Lewo
    'd': ord('d'),      # Prawo
    'w': ord('w'),      # Góra
    's': ord('s'),      # Dół
    'f': ord('f'),      # Strzał
    'enter': 13,        # Enter
    'up': 38,           # Strzałka góra
    'down': 40,         # Strzałka dół
    'p': ord('p')       # Play w menu
}


def perform_action(action: str) -> None:
    """
    Wykonaj akcję sterowania w grze.
    
    Args:
        action: Nazwa akcji do wykonania
        
    Raises:
        ValueError: Jeśli podano nieznaną akcję
    """
    try:
        if action == MENU_UP:
            logger.debug(f'[MENU] Naciśnięto: UP (kod: {KEYCODES["up"]})')
            pyautogui.keyDown('up')
            time.sleep(MENU_DELAY)
            pyautogui.keyUp('up')
            
        elif action == MENU_DOWN:
            logger.debug(f'[MENU] Naciśnięto: DOWN (kod: {KEYCODES["down"]})')
            pyautogui.keyDown('down')
            time.sleep(MENU_DELAY)
            pyautogui.keyUp('down')
            
        elif action == MENU_SELECT:
            logger.debug(f'[MENU] Naciśnięto: P (kod: {KEYCODES["p"]})')
            pyautogui.keyDown('p')
            time.sleep(MENU_DELAY)
            pyautogui.keyUp('p')
            time.sleep(MENU_DELAY)
            
            logger.debug(f'[MENU] Naciśnięto: ENTER (kod: {KEYCODES["enter"]})')
            pyautogui.keyDown('enter')
            time.sleep(MENU_DELAY)
            pyautogui.keyUp('enter')
            
        # Akcje gry
        elif action == MOVE_LEFT:
            logger.debug(f'[GRA] Naciśnięto: A (lewo) (kod: {KEYCODES["a"]})')
            pyautogui.keyDown('a')
            time.sleep(ACTION_DELAY)
            pyautogui.keyUp('a')
            
        elif action == MOVE_RIGHT:
            logger.debug(f'[GRA] Naciśnięto: D (prawo) (kod: {KEYCODES["d"]})')
            pyautogui.keyDown('d')
            time.sleep(ACTION_DELAY)
            pyautogui.keyUp('d')
            
        elif action == MOVE_UP:
            logger.debug(f'[GRA] Naciśnięto: W (góra) (kod: {KEYCODES["w"]})')
            pyautogui.keyDown('w')
            time.sleep(ACTION_DELAY)
            pyautogui.keyUp('w')
            
        elif action == MOVE_DOWN:
            logger.debug(f'[GRA] Naciśnięto: S (dół) (kod: {KEYCODES["s"]})')
            pyautogui.keyDown('s')
            time.sleep(ACTION_DELAY)
            pyautogui.keyUp('s')
            
        elif action in ['space', 'shoot']:
            logger.debug(f'[GRA] Naciśnięto: F (strzał) (kod: {KEYCODES["f"]})')
            pyautogui.keyDown('f')
            time.sleep(ACTION_DELAY)
            pyautogui.keyUp('f')
            
        else:
            logger.warning(f"Nieznana akcja: {action}")
            
    except Exception as e:
        logger.error(f"Błąd podczas wykonywania akcji '{action}': {e}")
        raise


def test_menu_keys() -> None:
    """
    Testuj klawisze menu - funkcja pomocnicza do debugowania.
    """
    logger.info("Testowanie klawiszy menu...")
    
    test_keys = [
        'up', 'down', 'left', 'right', 
        'w', 's', 'a', 'd', 
        'tab', '1', '2', '3', '4', 
        'numpad1', 'numpad2', 'numpad3', 'numpad4', 'numpad8'
    ]
    
    for key in test_keys:
        logger.info(f'Testuję klawisz: {key}')
        try:
            pyautogui.press(key)
            time.sleep(0.7)
        except Exception as e:
            logger.error(f"Błąd podczas testowania klawisza '{key}': {e}")


def check_ahk_availability() -> bool:
    """
    Sprawdź czy AutoHotkey jest dostępny w systemie.
    
    Returns:
        True jeśli AutoHotkey jest dostępny, False w przeciwnym razie
    """
    return os.path.exists(AHK_PATH)


def execute_ahk_script(script_path: str) -> Optional[subprocess.Popen]:
    """
    Wykonaj skrypt AutoHotkey.
    
    Args:
        script_path: Ścieżka do skryptu .ahk
        
    Returns:
        Obiekt subprocess.Popen lub None w przypadku błędu
    """
    if not check_ahk_availability():
        logger.warning("AutoHotkey nie jest dostępny w systemie")
        return None
        
    if not os.path.exists(script_path):
        logger.warning(f"Skrypt AutoHotkey nie istnieje: {script_path}")
        return None
        
    try:
        process = subprocess.Popen([AHK_PATH, script_path])
        logger.info(f"Uruchomiono skrypt AutoHotkey: {script_path}")
        return process
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania skryptu AutoHotkey: {e}")
        return None 