# -*- coding: utf-8 -*-
"""
Space Invaders AI Pro v3.0
==========================

Zaawansowany system sztucznej inteligencji do gry Space Invaders,
wykorzystujący Deep Q-Learning (DQN) z monitoringiem, testami i
zaawansowanymi algorytmami computer vision.

Autor: lonter100
Wersja: 3.0
Licencja: MIT

Opis:
    Profesjonalny system AI do automatycznej gry w Space Invaders.
    Wykorzystuje computer vision do wykrywania obiektów gry,
    Deep Q-Learning do podejmowania decyzji oraz
    zaawansowany system nagród i kar.
    Nowa wersja zawiera:
    - Zaawansowane algorytmy computer vision
    - Lepsze zarządzanie pamięcią i wydajnością
    - Rozbudowany system logowania i monitoringu
    - Automatyczne testy i walidację
    - Konfigurowalne parametry przez GUI
"""
import sys
import os
import time
import cv2
import numpy as np
import torch
import pygetwindow as gw
import logging
import argparse
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

# Dodaj ścieżkę do modułów
sys.path.append(str(Path(__file__).parent))

# Importy lokalnych modułów
from core.screen_utils import find_game_region, grab_screen, save_lives_region
from core.controller import perform_action
from core.logger import setup_logger, log_performance, log_game_event
from configs.config import (
    TEMPLATE_PLAYER_PATH, TEMPLATE_ENEMY_PATH, SCORE_REGION,
    LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY
)
from game_ai import GameAI

# Opcjonalne importy dla nowych modułów
try:
    from utils.performance_monitor import PerformanceMonitor
    from utils.config_manager import ConfigManager
    from utils.gui_manager import GUIManager
    NEW_MODULES_AVAILABLE = True
except ImportError:
    NEW_MODULES_AVAILABLE = False
    print("Uwaga: Niektóre zaawansowane moduły nie są dostępne")


class SpaceInvadersAI:
    """
    Główna klasa aplikacji Space Invaders AI Pro v3.0.
    
    Zawiera zaawansowane funkcje:
    - Automatyczne wykrywanie i konfiguracja gry
    - Zarządzanie wydajnością i pamięcią
    - System monitoringu i logowania
    - Interfejs GUI do konfiguracji
    - Automatyczne testy i walidację
    """
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Inicjalizacja głównej aplikacji AI.
        
        Args:
            config_path: Ścieżka do pliku konfiguracyjnego
        """
        self.config_path = config_path
        self.logger = setup_logger('space_invaders_ai_v3')
        
        # Inicjalizuj opcjonalne moduły
        if NEW_MODULES_AVAILABLE:
            self.performance_monitor = PerformanceMonitor()
            self.config_manager = ConfigManager(config_path)
            self.gui_manager = None
        else:
            self.performance_monitor = None
            self.config_manager = None
            self.gui_manager = None
        
        # Sprawdź wymagania systemowe
        self._check_system_requirements()
        
        # Inicjalizuj komponenty
        self._initialize_components()
        
        self.logger.info("Space Invaders AI Pro v3.0 zainicjalizowany pomyślnie")
    
    def _check_system_requirements(self) -> None:
        """Sprawdź wymagania systemowe i dostępność komponentów."""
        self.logger.info("Sprawdzanie wymagań systemowych...")
        
        # Sprawdź dostępność CUDA
        if torch.cuda.is_available():
            self.logger.info(f"CUDA dostępne: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.warning("CUDA niedostępne - używanie CPU")
        
        # Sprawdź dostępność OpenCV
        if cv2.__version__:
            self.logger.info(f"OpenCV wersja: {cv2.__version__}")
        
        # Sprawdź dostępność szablonów
        if os.path.exists(TEMPLATE_PLAYER_PATH):
            self.logger.info("Szablon gracza dostępny")
        else:
            self.logger.warning("Brak szablonu gracza - wykrywanie tylko po kolorze")
        
        if os.path.exists(TEMPLATE_ENEMY_PATH):
            self.logger.info("Szablon wroga dostępny")
        else:
            self.logger.warning("Brak szablonu wroga - wykrywanie tylko po kolorze")
    
    def _initialize_components(self) -> None:
        """Inicjalizuj komponenty aplikacji."""
        try:
            # Inicjalizuj GUI (opcjonalnie)
            if NEW_MODULES_AVAILABLE and self.config_manager and self.config_manager.get('gui.enable_gui', False):
                self.gui_manager = GUIManager()
                self.logger.info("GUI zainicjalizowane")
            
            # Sprawdź dostępność okna DOSBox
            self._check_dosbox_window()
            
        except Exception as e:
            self.logger.error(f"Błąd podczas inicjalizacji komponentów: {e}")
            raise
    
    def _check_dosbox_window(self) -> None:
        """Sprawdź dostępność okna DOSBox."""
        windows = gw.getWindowsWithTitle('DOSBox')
        if windows:
            win = windows[0]
            self.logger.info(f"Znaleziono okno DOSBox: {win.left}, {win.top}, {win.width}x{win.height}")
        else:
            self.logger.warning("Nie znaleziono okna DOSBox - używanie całego ekranu")
    
    @log_performance
    def run(self) -> None:
        """
        Uruchom główną pętlę aplikacji AI.
        
        Zawiera:
        - Automatyczne wykrywanie stanu gry
        - Zarządzanie wydajnością
        - Monitoring i logowanie
        - Obsługę błędów i wyjątków
        """
        self.logger.info("Uruchamianie Space Invaders AI Pro v3.0...")
        
        try:
            # Inicjalizuj AI
            game_ai = GameAI(self.config_path)
            
            # Uruchom monitoring wydajności
            if self.performance_monitor:
                self.performance_monitor.start()
            
            # Główna pętla
            self._main_loop(game_ai)
            
        except KeyboardInterrupt:
            self.logger.info("Przerwano przez użytkownika (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Krytyczny błąd aplikacji: {e}")
            raise
        finally:
            self._cleanup()
    
    def _main_loop(self, game_ai: GameAI) -> None:
        """Główna pętla aplikacji z zaawansowanym monitoringiem."""
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                frame_start = time.time()
                
                # Uruchom AI
                game_ai.run()
                frame_count += 1
                
                # Monitoring wydajności
                if self.performance_monitor:
                    frame_time = time.time() - frame_start
                    self.performance_monitor.update(frame_time)
                
                # Logowanie co 100 klatek
                if frame_count % 100 == 0:
                    fps = frame_count / (time.time() - start_time)
                    self.logger.info(f"FPS: {fps:.2f}, Frame: {frame_count}")
                
                # Sprawdź warunki zatrzymania
                if self._should_stop():
                    break
                    
            except Exception as e:
                self.logger.error(f"Błąd w głównej pętli: {e}")
                time.sleep(0.1)  # Krótka przerwa przed ponowną próbą
    
    def _should_stop(self) -> bool:
        """Sprawdź czy aplikacja powinna się zatrzymać."""
        # Sprawdź klawisz 'q' w oknie OpenCV
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        
        # Sprawdź warunki z GUI
        if self.gui_manager and self.gui_manager.should_stop():
            return True
        
        return False
    
    def _cleanup(self) -> None:
        """Wyczyść zasoby przed zakończeniem."""
        self.logger.info("Czyszczenie zasobów...")
        
        # Zatrzymaj monitoring
        if self.performance_monitor:
            self.performance_monitor.stop()
        
        # Zamknij GUI
        if self.gui_manager:
            self.gui_manager.cleanup()
        
        # Zamknij okna OpenCV
        cv2.destroyAllWindows()
        
        self.logger.info("Aplikacja zakończona pomyślnie")


def parse_arguments() -> argparse.Namespace:
    """Parsuj argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description='Space Invaders AI Pro v3.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python main.py                    # Uruchom z domyślną konfiguracją
  python main.py --config custom.yaml  # Użyj własnej konfiguracji
  python main.py --gui              # Uruchom z interfejsem GUI
  python main.py --test             # Uruchom w trybie testowym
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Ścieżka do pliku konfiguracyjnego'
    )
    
    parser.add_argument(
        '--gui', 
        action='store_true',
        help='Uruchom z interfejsem GUI'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Uruchom w trybie testowym'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Włącz tryb debugowania'
    )
    
    return parser.parse_args()


def main():
    """
    Główna funkcja uruchamiająca aplikację.
    
    Obsługuje:
    - Parsowanie argumentów wiersza poleceń
    - Inicjalizację aplikacji
    - Obsługę błędów i wyjątków
    - Logowanie i monitoring
    """
    try:
        # Parsuj argumenty
        args = parse_arguments()
        
        # Ustaw poziom logowania
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Utwórz i uruchom aplikację
        app = SpaceInvadersAI(args.config)
        
        # Ustaw flagi z argumentów
        if args.gui and NEW_MODULES_AVAILABLE and app.config_manager:
            app.config_manager.set('gui.enable_gui', True)
        if args.test and app.config_manager:
            app.config_manager.set('testing.test_mode', True)
        
        # Uruchom aplikację
        app.run()
        
    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika")
    except Exception as e:
        print(f"Krytyczny błąd: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 