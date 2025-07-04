# -*- coding: utf-8 -*-
"""
Moduł zarządzania GUI Space Invaders AI Pro v3.0
=================================================

Zaawansowany interfejs graficzny dla aplikacji AI.

Autor: lonter100
Wersja: 3.0
Licencja: MIT

Opis:
    Zawiera funkcje do:
    - Interfejsu graficznego w czasie rzeczywistym
    - Kontroli parametrów AI
    - Wizualizacji statystyk
    - Monitorowania wydajności
    - Zarządzania konfiguracją
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from typing import Dict, Any, Optional, Callable
import logging
from pathlib import Path


class GUIManager:
    """
    Zaawansowany menedżer GUI dla aplikacji AI.
    
    Funkcje:
    - Interfejs w czasie rzeczywistym
    - Kontrola parametrów AI
    - Wizualizacja statystyk
    - Monitoring wydajności
    - Zarządzanie konfiguracją
    """
    
    def __init__(self):
        """Inicjalizacja menedżera GUI."""
        self.logger = logging.getLogger(__name__)
        self.root = None
        self.is_running = False
        self.update_callbacks = {}
        
        # Statystyki do wyświetlenia
        self.stats = {}
        
        self.logger.info("Menadżer GUI zainicjalizowany")
    
    def create_gui(self) -> None:
        """Utwórz główne okno GUI."""
        try:
            self.root = tk.Tk()
            self.root.title("Space Invaders AI Pro v3.0")
            self.root.geometry("1000x700")
            self.root.configure(bg='#2b2b2b')
            
            # Style
            style = ttk.Style()
            style.theme_use('clam')
            style.configure('TFrame', background='#2b2b2b')
            style.configure('TLabel', background='#2b2b2b', foreground='white')
            style.configure('TButton', background='#4a4a4a', foreground='white')
            
            self._create_widgets()
            self._setup_layout()
            
            self.is_running = True
            self.logger.info("GUI utworzone pomyślnie")
            
        except Exception as e:
            self.logger.error(f"Błąd podczas tworzenia GUI: {e}")
            raise
    
    def _create_widgets(self) -> None:
        """Utwórz widgety GUI."""
        # Główny kontener
        self.main_frame = ttk.Frame(self.root)
        
        # Notebook dla zakładek
        self.notebook = ttk.Notebook(self.main_frame)
        
        # Zakładka główna
        self.main_tab = ttk.Frame(self.notebook)
        self._create_main_tab()
        
        # Zakładka AI
        self.ai_tab = ttk.Frame(self.notebook)
        self._create_ai_tab()
        
        # Zakładka wydajności
        self.performance_tab = ttk.Frame(self.notebook)
        self._create_performance_tab()
        
        # Zakładka konfiguracji
        self.config_tab = ttk.Frame(self.notebook)
        self._create_config_tab()
        
        # Dodaj zakładki
        self.notebook.add(self.main_tab, text="Główna")
        self.notebook.add(self.ai_tab, text="AI")
        self.notebook.add(self.performance_tab, text="Wydajność")
        self.notebook.add(self.config_tab, text="Konfiguracja")
    
    def _create_main_tab(self) -> None:
        """Utwórz zakładkę główną."""
        # Status aplikacji
        status_frame = ttk.LabelFrame(self.main_tab, text="Status aplikacji", padding=10)
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Gotowy")
        self.status_label.pack()
        
        # Kontrolki
        control_frame = ttk.LabelFrame(self.main_tab, text="Kontrolki", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start AI", command=self._on_start)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop AI", command=self._on_stop)
        self.stop_button.pack(side='left', padx=5)
        
        self.pause_button = ttk.Button(control_frame, text="Pauza", command=self._on_pause)
        self.pause_button.pack(side='left', padx=5)
        
        # Statystyki gry
        game_frame = ttk.LabelFrame(self.main_tab, text="Statystyki gry", padding=10)
        game_frame.pack(fill='x', padx=10, pady=5)
        
        self.score_label = ttk.Label(game_frame, text="Wynik: 0")
        self.score_label.pack()
        
        self.lives_label = ttk.Label(game_frame, text="Życia: 3")
        self.lives_label.pack()
        
        self.round_label = ttk.Label(game_frame, text="Runda: 1")
        self.round_label.pack()
    
    def _create_ai_tab(self) -> None:
        """Utwórz zakładkę AI."""
        # Parametry AI
        ai_frame = ttk.LabelFrame(self.ai_tab, text="Parametry AI", padding=10)
        ai_frame.pack(fill='x', padx=10, pady=5)
        
        # Learning rate
        ttk.Label(ai_frame, text="Learning Rate:").grid(row=0, column=0, sticky='w')
        self.lr_var = tk.DoubleVar(value=1e-4)
        self.lr_scale = ttk.Scale(ai_frame, from_=1e-6, to=1e-2, variable=self.lr_var, 
                                 orient='horizontal', length=200)
        self.lr_scale.grid(row=0, column=1, padx=5)
        self.lr_label = ttk.Label(ai_frame, text="1e-4")
        self.lr_label.grid(row=0, column=2)
        
        # Epsilon
        ttk.Label(ai_frame, text="Epsilon:").grid(row=1, column=0, sticky='w')
        self.epsilon_var = tk.DoubleVar(value=0.1)
        self.epsilon_scale = ttk.Scale(ai_frame, from_=0.0, to=1.0, variable=self.epsilon_var,
                                      orient='horizontal', length=200)
        self.epsilon_scale.grid(row=1, column=1, padx=5)
        self.epsilon_label = ttk.Label(ai_frame, text="0.1")
        self.epsilon_label.grid(row=1, column=2)
        
        # Gamma
        ttk.Label(ai_frame, text="Gamma:").grid(row=2, column=0, sticky='w')
        self.gamma_var = tk.DoubleVar(value=0.99)
        self.gamma_scale = ttk.Scale(ai_frame, from_=0.8, to=0.999, variable=self.gamma_var,
                                     orient='horizontal', length=200)
        self.gamma_scale.grid(row=2, column=1, padx=5)
        self.gamma_label = ttk.Label(ai_frame, text="0.99")
        self.gamma_label.grid(row=2, column=2)
        
        # Przyciski
        button_frame = ttk.Frame(ai_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Zapisz parametry", 
                  command=self._save_ai_params).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Resetuj parametry", 
                  command=self._reset_ai_params).pack(side='left', padx=5)
        
        # Statystyki AI
        ai_stats_frame = ttk.LabelFrame(self.ai_tab, text="Statystyki AI", padding=10)
        ai_stats_frame.pack(fill='x', padx=10, pady=5)
        
        self.ai_accuracy_label = ttk.Label(ai_stats_frame, text="Dokładność: 0%")
        self.ai_accuracy_label.pack()
        
        self.ai_reward_label = ttk.Label(ai_stats_frame, text="Średnia nagroda: 0.0")
        self.ai_reward_label.pack()
        
        self.ai_decisions_label = ttk.Label(ai_stats_frame, text="Decyzje: 0")
        self.ai_decisions_label.pack()
    
    def _create_performance_tab(self) -> None:
        """Utwórz zakładkę wydajności."""
        # FPS
        fps_frame = ttk.LabelFrame(self.performance_tab, text="Wydajność", padding=10)
        fps_frame.pack(fill='x', padx=10, pady=5)
        
        self.fps_label = ttk.Label(fps_frame, text="FPS: 0")
        self.fps_label.pack()
        
        self.frame_time_label = ttk.Label(fps_frame, text="Czas klatki: 0ms")
        self.frame_time_label.pack()
        
        # Zasoby systemowe
        resources_frame = ttk.LabelFrame(self.performance_tab, text="Zasoby systemowe", padding=10)
        resources_frame.pack(fill='x', padx=10, pady=5)
        
        self.cpu_label = ttk.Label(resources_frame, text="CPU: 0%")
        self.cpu_label.pack()
        
        self.memory_label = ttk.Label(resources_frame, text="Pamięć: 0%")
        self.memory_label.pack()
        
        self.gpu_label = ttk.Label(resources_frame, text="GPU: N/A")
        self.gpu_label.pack()
        
        # Alerty
        alerts_frame = ttk.LabelFrame(self.performance_tab, text="Alerty", padding=10)
        alerts_frame.pack(fill='x', padx=10, pady=5)
        
        self.alerts_text = tk.Text(alerts_frame, height=5, width=50, bg='#3c3c3c', fg='white')
        self.alerts_text.pack()
    
    def _create_config_tab(self) -> None:
        """Utwórz zakładkę konfiguracji."""
        # Zarządzanie konfiguracją
        config_frame = ttk.LabelFrame(self.config_tab, text="Zarządzanie konfiguracją", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(config_frame, text="Załaduj konfigurację", 
                  command=self._load_config).pack(side='left', padx=5)
        ttk.Button(config_frame, text="Zapisz konfigurację", 
                  command=self._save_config).pack(side='left', padx=5)
        ttk.Button(config_frame, text="Eksportuj konfigurację", 
                  command=self._export_config).pack(side='left', padx=5)
        ttk.Button(config_frame, text="Importuj konfigurację", 
                  command=self._import_config).pack(side='left', padx=5)
        
        # Backup
        backup_frame = ttk.LabelFrame(self.config_tab, text="Backup", padding=10)
        backup_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(backup_frame, text="Utwórz backup", 
                  command=self._create_backup).pack(side='left', padx=5)
        ttk.Button(backup_frame, text="Przywróć backup", 
                  command=self._restore_backup).pack(side='left', padx=5)
        
        # Lista backupów
        self.backup_listbox = tk.Listbox(backup_frame, height=5, bg='#3c3c3c', fg='white')
        self.backup_listbox.pack(fill='x', pady=5)
    
    def _setup_layout(self) -> None:
        """Skonfiguruj układ GUI."""
        self.main_frame.pack(fill='both', expand=True)
        self.notebook.pack(fill='both', expand=True)
    
    def run(self) -> None:
        """Uruchom GUI."""
        if self.root:
            self.root.mainloop()
    
    def update_stats(self, stats: Dict[str, Any]) -> None:
        """Zaktualizuj statystyki w GUI."""
        self.stats = stats
        
        # Aktualizuj etykiety
        if 'score' in stats:
            self.score_label.config(text=f"Wynik: {stats['score']}")
        
        if 'lives' in stats:
            self.lives_label.config(text=f"Życia: {stats['lives']}")
        
        if 'round' in stats:
            self.round_label.config(text=f"Runda: {stats['round']}")
        
        if 'fps' in stats:
            self.fps_label.config(text=f"FPS: {stats['fps']:.1f}")
        
        if 'frame_time' in stats:
            self.frame_time_label.config(text=f"Czas klatki: {stats['frame_time']*1000:.1f}ms")
        
        if 'cpu_usage' in stats:
            self.cpu_label.config(text=f"CPU: {stats['cpu_usage']:.1f}%")
        
        if 'memory_usage' in stats:
            self.memory_label.config(text=f"Pamięć: {stats['memory_usage']:.1f}%")
        
        if 'ai_accuracy' in stats:
            self.ai_accuracy_label.config(text=f"Dokładność: {stats['ai_accuracy']*100:.1f}%")
        
        if 'avg_reward' in stats:
            self.ai_reward_label.config(text=f"Średnia nagroda: {stats['avg_reward']:.2f}")
    
    def _on_start(self) -> None:
        """Obsługa przycisku Start."""
        self.status_label.config(text="AI uruchomione")
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Wywołaj callback
        if 'start' in self.update_callbacks:
            self.update_callbacks['start']()
    
    def _on_stop(self) -> None:
        """Obsługa przycisku Stop."""
        self.status_label.config(text="AI zatrzymane")
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        # Wywołaj callback
        if 'stop' in self.update_callbacks:
            self.update_callbacks['stop']()
    
    def _on_pause(self) -> None:
        """Obsługa przycisku Pauza."""
        # Wywołaj callback
        if 'pause' in self.update_callbacks:
            self.update_callbacks['pause']()
    
    def _save_ai_params(self) -> None:
        """Zapisz parametry AI."""
        # Wywołaj callback
        if 'save_ai_params' in self.update_callbacks:
            params = {
                'learning_rate': self.lr_var.get(),
                'epsilon': self.epsilon_var.get(),
                'gamma': self.gamma_var.get(),
            }
            self.update_callbacks['save_ai_params'](params)
    
    def _reset_ai_params(self) -> None:
        """Resetuj parametry AI."""
        self.lr_var.set(1e-4)
        self.epsilon_var.set(0.1)
        self.gamma_var.set(0.99)
    
    def _load_config(self) -> None:
        """Załaduj konfigurację."""
        filename = filedialog.askopenfilename(
            title="Wybierz plik konfiguracyjny",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            # Wywołaj callback
            if 'load_config' in self.update_callbacks:
                self.update_callbacks['load_config'](filename)
    
    def _save_config(self) -> None:
        """Zapisz konfigurację."""
        # Wywołaj callback
        if 'save_config' in self.update_callbacks:
            self.update_callbacks['save_config']()
    
    def _export_config(self) -> None:
        """Eksportuj konfigurację."""
        filename = filedialog.asksaveasfilename(
            title="Zapisz konfigurację",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            # Wywołaj callback
            if 'export_config' in self.update_callbacks:
                self.update_callbacks['export_config'](filename)
    
    def _import_config(self) -> None:
        """Importuj konfigurację."""
        filename = filedialog.askopenfilename(
            title="Wybierz plik do importu",
            filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            # Wywołaj callback
            if 'import_config' in self.update_callbacks:
                self.update_callbacks['import_config'](filename)
    
    def _create_backup(self) -> None:
        """Utwórz backup."""
        # Wywołaj callback
        if 'create_backup' in self.update_callbacks:
            self.update_callbacks['create_backup']()
    
    def _restore_backup(self) -> None:
        """Przywróć backup."""
        selection = self.backup_listbox.curselection()
        if selection:
            backup_name = self.backup_listbox.get(selection[0])
            # Wywołaj callback
            if 'restore_backup' in self.update_callbacks:
                self.update_callbacks['restore_backup'](backup_name)
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Zarejestruj callback dla zdarzenia."""
        self.update_callbacks[event] = callback
    
    def should_stop(self) -> bool:
        """Sprawdź czy GUI zostało zamknięte."""
        return not self.is_running
    
    def cleanup(self) -> None:
        """Wyczyść zasoby GUI."""
        self.is_running = False
        if self.root:
            self.root.quit()
            self.root.destroy() 