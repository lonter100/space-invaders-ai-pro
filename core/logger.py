# -*- coding: utf-8 -*-
"""
Moduł logger Space Invaders AI Pro
==================================

Konfiguracja systemu logowania dla aplikacji AI.

Autor: lonter100
Wersja: 2.0
Licencja: MIT

Opis:
    Zawiera funkcje do konfiguracji loggera z różnymi poziomami
    logowania, rotacją plików i formatowaniem wiadomości.
"""
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str = 'space_invaders_ai', 
                log_file: str = 'log_ai.txt',
                level: int = logging.INFO) -> logging.Logger:
    """
    Skonfiguruj logger dla aplikacji Space Invaders AI.
    
    Args:
        name: Nazwa loggera
        log_file: Ścieżka do pliku logów
        level: Poziom logowania
        
    Returns:
        Skonfigurowany logger
    """
    # Utwórz logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Wyczyść istniejące handlery
    logger.handlers.clear()
    
    # Format wiadomości
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler do pliku z rotacją
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Błąd podczas konfiguracji handlera pliku: {e}")
    
    # Handler do konsoli
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = 'space_invaders_ai') -> logging.Logger:
    """
    Pobierz istniejący logger lub utwórz nowy.
    
    Args:
        name: Nazwa loggera
        
    Returns:
        Logger
    """
    return logging.getLogger(name)


def log_performance(func):
    """
    Dekorator do logowania wydajności funkcji.
    
    Args:
        func: Funkcja do dekorowania
        
    Returns:
        Zdekorowana funkcja
    """
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.debug(f"{func.__name__} wykonana w {duration:.3f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"{func.__name__} zakończona błędem po {duration:.3f}s: {e}")
            raise
    
    return wrapper


def log_game_event(event_type: str, details: Optional[dict] = None):
    """
    Zaloguj zdarzenie gry.
    
    Args:
        event_type: Typ zdarzenia (np. 'score_change', 'life_lost')
        details: Dodatkowe szczegóły zdarzenia
    """
    logger = get_logger()
    
    if details:
        logger.info(f"GAME_EVENT: {event_type} - {details}")
    else:
        logger.info(f"GAME_EVENT: {event_type}")


def log_ai_decision(action: str, confidence: float = None, 
                   state_info: Optional[dict] = None):
    """
    Zaloguj decyzję AI.
    
    Args:
        action: Wykonana akcja
        confidence: Poziom pewności (0.0-1.0)
        state_info: Informacje o stanie gry
    """
    logger = get_logger()
    
    msg = f"AI_DECISION: {action}"
    if confidence is not None:
        msg += f" (confidence: {confidence:.3f})"
    if state_info:
        msg += f" - {state_info}"
        
    logger.debug(msg) 