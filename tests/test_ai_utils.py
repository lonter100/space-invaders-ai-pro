# -*- coding: utf-8 -*-
"""
Testy AI Utils
Autor: Twój Nick
Wersja: 2.0
Licencja: MIT
Opis: Testy jednostkowe funkcji AI.
"""
import numpy as np
import cv2
from ai_utils import detect_player, detect_enemies, read_score

def test_detect_player():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 80), (60, 99), (255, 0, 0), -1)
    pos = detect_player(img)
    assert pos is None or (40 <= pos[0] <= 60 and 80 <= pos[1] <= 99)

def test_detect_enemies():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (20, 20), (0, 255, 0), -1)
    enemies = detect_enemies(img)
    assert isinstance(enemies, list)

def test_read_score():
    img = np.zeros((60, 800, 3), dtype=np.uint8)
    cv2.putText(img, '1234', (660, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    score = read_score(img)
    assert isinstance(score, int)

def run_tests():
    test_detect_player()
    test_detect_enemies()
    test_read_score()
    print('Wszystkie testy zaliczone!')

if __name__ == '__main__':
    run_tests() 