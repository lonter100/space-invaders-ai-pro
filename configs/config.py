# -*- coding: utf-8 -*-
"""
Konfiguracja Space Invaders AI Pro
==================================

Stałe i parametry konfiguracyjne dla systemu AI.

Autor: lonter100
Wersja: 2.0
Licencja: MIT

Opis:
    Zawiera wszystkie parametry konfiguracyjne potrzebne do działania
    systemu AI, w tym kolory do wykrywania obiektów, regiony ekranu,
    progi i ścieżki do szablonów.
"""

# =============================================================================
# PARAMETRY WYKRYWANIA OBIEKTÓW
# =============================================================================

# Kolory gracza (BGR format)
PLAYER_COLOR_LOWER = [0, 0, 200]      # Dolny próg koloru gracza
PLAYER_COLOR_UPPER = [100, 100, 255]  # Górny próg koloru gracza

# Kolory wrogów (BGR format)
ENEMY_COLOR_LOWER = [0, 200, 0]       # Dolny próg koloru wrogów
ENEMY_COLOR_UPPER = [100, 255, 100]   # Górny próg koloru wrogów

# =============================================================================
# REGIONY EKRANU
# =============================================================================

# Region wyniku (y1, y2, x1, x2) - pole z cyframi
SCORE_REGION = (55, 95, 22, 170)

# Minimalny rozmiar okna gry
MIN_GAME_WINDOW_SIZE = (200, 200)

# =============================================================================
# PARAMETRY TEMPLATE MATCHING
# =============================================================================

# Próg dopasowania szablonu (0.0-1.0)
TEMPLATE_MATCH_THRESHOLD = 0.7

# Ścieżki do szablonów obiektów
TEMPLATE_PLAYER_PATH = 'player_template.png'
TEMPLATE_ENEMY_PATH = 'enemy_template.png'

# =============================================================================
# PARAMETRY OCR
# =============================================================================

# Próg dla OCR (0-255)
SCORE_THRESHOLD = 150

# =============================================================================
# PARAMETRY STEROWANIA
# =============================================================================

# Opóźnienia między akcjami (w sekundach)
ACTION_DELAY = 0.01
MENU_DELAY = 0.05

# =============================================================================
# PARAMETRY AI
# =============================================================================

# Parametry DQN
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

# Rozmiary buforów
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32

# Częstotliwość aktualizacji target network
TARGET_UPDATE_FREQ = 1000 