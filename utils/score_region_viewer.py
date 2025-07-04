# -*- coding: utf-8 -*-
"""
Score Region Viewer
Autor: Twój Nick
Wersja: 2.0
Licencja: MIT
Opis: Narzędzie do podglądu regionu wyniku na ekranie gry.
"""
import cv2
import numpy as np
import mss
import time
from config import SCORE_REGION
from screen_utils import find_game_region

def main():
    print('Podgląd SCORE_REGION na tle całego ekranu. Zamknij okno, by zakończyć.')
    game_region = find_game_region()
    with mss.mss() as sct:
        while True:
            # Pobierz cały ekran
            monitor = sct.monitors[1]
            screen = np.array(sct.grab(monitor))
            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
            # Zaznacz okno gry
            xg, yg, wg, hg = game_region['left'], game_region['top'], game_region['width'], game_region['height']
            cv2.rectangle(screen_bgr, (xg, yg), (xg+wg, yg+hg), (0,255,255), 2)
            # Zaznacz SCORE_REGION względem całego ekranu
            y1, y2, x1, x2 = SCORE_REGION
            xs, ys = xg + x1, yg + y1
            xe, ye = xg + x2, yg + y2
            cv2.rectangle(screen_bgr, (xs, ys), (xe, ye), (0,0,255), 2)
            cv2.putText(screen_bgr, 'SCORE_REGION', (xs, ys-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(screen_bgr, 'GAME_REGION', (xg, yg-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            # Wytnij SCORE_REGION z całego ekranu
            score_crop = screen_bgr[ys:ye, xs:xe]
            cv2.imshow('Ekran z regionami', screen_bgr)
            cv2.imshow('SCORE_REGION', score_crop)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 