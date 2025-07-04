# -*- coding: utf-8 -*-
"""
Moduł zarządzania konfiguracją Space Invaders AI Pro v3.0
==========================================================

Zaawansowany system zarządzania konfiguracją aplikacji.

Autor: lonter100
Wersja: 3.0
Licencja: MIT

Opis:
    Zawiera funkcje do:
    - Ładowania i zapisywania konfiguracji
    - Walidacji parametrów
    - Dynamicznej zmiany ustawień
    - Backup i restore konfiguracji
    - Profili konfiguracyjnych
"""
import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import shutil


class ConfigManager:
    """
    Zaawansowany menedżer konfiguracji dla aplikacji AI.
    
    Funkcje:
    - Ładowanie/zapisywanie konfiguracji YAML/JSON
    - Walidacja parametrów
    - Dynamiczna zmiana ustawień
    - Backup i restore
    - Profile konfiguracyjne
    """
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Inicjalizacja menedżera konfiguracji.
        
        Args:
            config_path: Ścieżka do głównego pliku konfiguracyjnego
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        self.backup_dir = self.config_dir / 'backups'
        
        # Utwórz katalogi jeśli nie istnieją
        self.backup_dir.mkdir(exist_ok=True)
        
        # Konfiguracja
        self.config = {}
        self.default_config = self._get_default_config()
        
        # Historia zmian
        self.change_history = []
        
        # Załaduj konfigurację
        self.load_config()
        
        self.logger.info(f"Menadżer konfiguracji zainicjalizowany: {config_path}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Pobierz domyślną konfigurację."""
        return {
            'ai': {
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_min': 0.05,
                'epsilon_decay': 0.995,
                'batch_size': 32,
                'buffer_size': 10000,
                'target_update_freq': 1000,
                'device': 'auto',  # 'auto', 'cpu', 'cuda'
            },
            'vision': {
                'player_color_lower': [0, 0, 200],
                'player_color_upper': [100, 100, 255],
                'enemy_color_lower': [0, 200, 0],
                'enemy_color_upper': [100, 255, 100],
                'score_region': [55, 95, 22, 170],
                'score_threshold': 150,
                'template_match_threshold': 0.7,
            },
            'control': {
                'action_delay': 0.01,
                'menu_delay': 0.05,
                'key_mappings': {
                    'left': 'a',
                    'right': 'd',
                    'up': 'w',
                    'down': 's',
                    'shoot': 'f',
                    'menu_select': 'p',
                },
            },
            'performance': {
                'enable_monitoring': True,
                'fps_threshold': 30.0,
                'memory_threshold': 80.0,
                'cpu_threshold': 90.0,
                'auto_save_interval': 300,  # sekundy
            },
            'logging': {
                'level': 'INFO',
                'file_enabled': True,
                'console_enabled': True,
                'max_file_size': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5,
            },
            'gui': {
                'enable_gui': False,
                'window_width': 800,
                'window_height': 600,
                'theme': 'dark',
            },
            'testing': {
                'test_mode': False,
                'headless_mode': False,
                'save_debug_images': False,
                'log_ai_decisions': True,
            },
        }
    
    def load_config(self) -> None:
        """Załaduj konfigurację z pliku."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f) or {}
                
                # Połącz z domyślną konfiguracją
                self.config = self._merge_configs(self.default_config, file_config)
                self.logger.info(f"Konfiguracja załadowana: {self.config_path}")
            else:
                # Użyj domyślnej konfiguracji
                self.config = self.default_config.copy()
                self.save_config()  # Zapisz domyślną konfigurację
                self.logger.info("Utworzono domyślną konfigurację")
                
        except Exception as e:
            self.logger.error(f"Błąd podczas ładowania konfiguracji: {e}")
            self.config = self.default_config.copy()
    
    def save_config(self) -> None:
        """Zapisz konfigurację do pliku."""
        try:
            # Utwórz backup przed zapisem
            self._create_backup()
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Konfiguracja zapisana: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania konfiguracji: {e}")
    
    def _merge_configs(self, default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
        """Połącz konfiguracje rekurencyjnie."""
        result = default.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Pobierz wartość z konfiguracji.
        
        Args:
            key: Klucz w formacie 'section.subsection.key'
            default: Wartość domyślna jeśli klucz nie istnieje
            
        Returns:
            Wartość z konfiguracji
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Ustaw wartość w konfiguracji.
        
        Args:
            key: Klucz w formacie 'section.subsection.key'
            value: Nowa wartość
        """
        keys = key.split('.')
        config = self.config
        
        # Przejdź do ostatniego poziomu
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Zapisz zmianę
        old_value = config.get(keys[-1])
        config[keys[-1]] = value
        
        # Dodaj do historii
        self.change_history.append({
            'timestamp': datetime.now(),
            'key': key,
            'old_value': old_value,
            'new_value': value,
        })
        
        self.logger.debug(f"Zmieniono konfigurację: {key} = {value}")
    
    def validate_config(self) -> List[str]:
        """
        Zwaliduj konfigurację.
        
        Returns:
            Lista błędów walidacji
        """
        errors = []
        
        # Sprawdź wymagane sekcje
        required_sections = ['ai', 'vision', 'control', 'performance']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Brakuje sekcji: {section}")
        
        # Sprawdź wartości AI
        ai_config = self.config.get('ai', {})
        if ai_config.get('learning_rate', 0) <= 0:
            errors.append("Learning rate musi być większe od 0")
        
        if not (0 < ai_config.get('gamma', 0) < 1):
            errors.append("Gamma musi być między 0 a 1")
        
        # Sprawdź kolory
        vision_config = self.config.get('vision', {})
        for color_key in ['player_color_lower', 'player_color_upper', 
                         'enemy_color_lower', 'enemy_color_upper']:
            color = vision_config.get(color_key, [])
            if len(color) != 3 or not all(0 <= c <= 255 for c in color):
                errors.append(f"Nieprawidłowy format koloru: {color_key}")
        
        # Sprawdź regiony
        score_region = vision_config.get('score_region', [])
        if len(score_region) != 4:
            errors.append("Score region musi mieć 4 wartości")
        
        return errors
    
    def _create_backup(self) -> None:
        """Utwórz backup konfiguracji."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"config_backup_{timestamp}.yaml"
            
            if self.config_path.exists():
                shutil.copy2(self.config_path, backup_path)
                
                # Usuń stare backupy (zostaw max 10)
                backups = sorted(self.backup_dir.glob('config_backup_*.yaml'))
                if len(backups) > 10:
                    for backup in backups[:-10]:
                        backup.unlink()
            
        except Exception as e:
            self.logger.warning(f"Błąd podczas tworzenia backupu: {e}")
    
    def restore_backup(self, backup_name: str) -> bool:
        """
        Przywróć konfigurację z backupu.
        
        Args:
            backup_name: Nazwa pliku backupu
            
        Returns:
            True jeśli przywrócono pomyślnie
        """
        try:
            backup_path = self.backup_dir / backup_name
            if backup_path.exists():
                shutil.copy2(backup_path, self.config_path)
                self.load_config()
                self.logger.info(f"Przywiedziono konfigurację z: {backup_name}")
                return True
            else:
                self.logger.error(f"Backup nie istnieje: {backup_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Błąd podczas przywracania backupu: {e}")
            return False
    
    def get_backups(self) -> List[str]:
        """Pobierz listę dostępnych backupów."""
        backups = []
        for backup_file in self.backup_dir.glob('config_backup_*.yaml'):
            backups.append(backup_file.name)
        return sorted(backups, reverse=True)
    
    def export_config(self, format: str = 'yaml') -> str:
        """
        Eksportuj konfigurację do stringa.
        
        Args:
            format: Format eksportu ('yaml' lub 'json')
            
        Returns:
            Konfiguracja jako string
        """
        try:
            if format.lower() == 'json':
                return json.dumps(self.config, indent=2, ensure_ascii=False)
            else:
                return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.logger.error(f"Błąd podczas eksportu konfiguracji: {e}")
            return ""
    
    def import_config(self, config_data: str, format: str = 'yaml') -> bool:
        """
        Zaimportuj konfigurację ze stringa.
        
        Args:
            config_data: Dane konfiguracji jako string
            format: Format importu ('yaml' lub 'json')
            
        Returns:
            True jeśli zaimportowano pomyślnie
        """
        try:
            if format.lower() == 'json':
                imported_config = json.loads(config_data)
            else:
                imported_config = yaml.safe_load(config_data)
            
            # Połącz z aktualną konfiguracją
            self.config = self._merge_configs(self.config, imported_config)
            
            # Zwaliduj
            errors = self.validate_config()
            if errors:
                self.logger.error(f"Błędy walidacji: {errors}")
                return False
            
            self.save_config()
            self.logger.info("Konfiguracja zaimportowana pomyślnie")
            return True
            
        except Exception as e:
            self.logger.error(f"Błąd podczas importu konfiguracji: {e}")
            return False
    
    def get_change_history(self) -> List[Dict[str, Any]]:
        """Pobierz historię zmian konfiguracji."""
        return self.change_history.copy()
    
    def reset_to_defaults(self) -> None:
        """Zresetuj konfigurację do wartości domyślnych."""
        self.config = self.default_config.copy()
        self.save_config()
        self.logger.info("Konfiguracja zresetowana do wartości domyślnych") 