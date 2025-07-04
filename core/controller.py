# -*- coding: utf-8 -*-
"""
Moduł sterowania
Autor: Twój Nick
Wersja: 2.0
Licencja: MIT
Opis: Obsługa sterowania ruchem i strzałem w grze.
"""
import pyautogui
import time
import keyboard
import subprocess
import os

MOVE_LEFT = 'left'
MOVE_RIGHT = 'right'
MOVE_UP = 'up'
MOVE_DOWN = 'down'
SHOOT = 'space'
MENU_SELECT = 'menu_select'

# Ścieżka do AutoHotkey.exe (v2)
AHK_PATH = r"C:\Program Files\AutoHotkey\v2\AutoHotkey.exe"
# Ścieżka do skryptu shoot.ahk (możesz trzymać w katalogu projektu)
AHK_SCRIPT = os.path.abspath("shoot.ahk")

# Mapowanie nazw klawiszy na kody Windows/ASCII
KEYCODES = {
    'a': ord('a'),
    'd': ord('d'),
    'w': ord('w'),
    's': ord('s'),
    'f': ord('f'),
    'enter': 13,
    'up': 38,
    'down': 40
}

def perform_action(action):
    # Akcje menu
    if action == 'menu_up':
        print(f'[MENU] Naciśnięto: UP (kod: {KEYCODES["up"]})')
        pyautogui.keyDown('up')
        time.sleep(0.01)
        pyautogui.keyUp('up')
    elif action == 'menu_down':
        print(f'[MENU] Naciśnięto: DOWN (kod: {KEYCODES["down"]})')
        pyautogui.keyDown('down')
        time.sleep(0.01)
        pyautogui.keyUp('down')
    elif action == 'menu_select':
        print(f'[MENU] Naciśnięto: P (kod: {KEYCODES.get("p", ord("p"))})')
        pyautogui.keyDown('p')
        time.sleep(0.01)
        pyautogui.keyUp('p')
        time.sleep(0.01)
        print(f'[MENU] Naciśnięto: ENTER (kod: {KEYCODES["enter"]})')
        pyautogui.keyDown('enter')
        time.sleep(0.01)
        pyautogui.keyUp('enter')
    # Akcje gry
    elif action == 'left':
        print(f'[GRA] Naciśnięto: A (lewo) (kod: {KEYCODES["a"]})')
        pyautogui.keyDown('a')
        time.sleep(0.01)
        pyautogui.keyUp('a')
    elif action == 'right':
        print(f'[GRA] Naciśnięto: D (prawo) (kod: {KEYCODES["d"]})')
        pyautogui.keyDown('d')
        time.sleep(0.01)
        pyautogui.keyUp('d')
    elif action == 'up':
        print(f'[GRA] Naciśnięto: W (góra) (kod: {KEYCODES["w"]})')
        pyautogui.keyDown('w')
        time.sleep(0.01)
        pyautogui.keyUp('w')
    elif action == 'down':
        print(f'[GRA] Naciśnięto: S (dół) (kod: {KEYCODES["s"]})')
        pyautogui.keyDown('s')
        time.sleep(0.01)
        pyautogui.keyUp('s')
    elif action == 'space' or action == 'shoot':
        print(f'[GRA] Naciśnięto: F (strzał) (kod: {KEYCODES["f"]})')
        pyautogui.keyDown('f')
        time.sleep(0.01)
        pyautogui.keyUp('f')
    # else: nic nie rób dla nieznanych akcji

def test_menu_keys():
    import pyautogui
    import time
    test_keys = ['up', 'down', 'left', 'right', 'w', 's', 'a', 'd', 'tab', '1', '2', '3', '4', 'numpad1', 'numpad2', 'numpad3', 'numpad4', 'numpad8']
    for key in test_keys:
        print(f'Testuję klawisz: {key}')
        pyautogui.press(key)
        time.sleep(0.7) 