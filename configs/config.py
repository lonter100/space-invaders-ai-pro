# -*- coding: utf-8 -*-
"""
Konfiguracja Space Invaders AI
Autor: Twój Nick
Wersja: 2.0
Licencja: MIT
Opis: Stałe i parametry konfiguracyjne.
"""
# Parametry kolorów i progów do wykrywania obiektów
PLAYER_COLOR_LOWER = [0, 0, 200]
PLAYER_COLOR_UPPER = [100, 100, 255]
ENEMY_COLOR_LOWER = [0, 200, 0]
ENEMY_COLOR_UPPER = [100, 255, 100]
# Region wyniku przesunięty wyżej (pole z cyframi)
SCORE_REGION = (55, 95, 22, 170)  # y1, y2, x1, x2
SCORE_THRESHOLD = 150
MIN_GAME_WINDOW_SIZE = (200, 200)
# Parametry template matching
TEMPLATE_MATCH_THRESHOLD = 0.7
TEMPLATE_PLAYER_PATH = 'player_template.png'
TEMPLATE_ENEMY_PATH = 'enemy_template.png' 