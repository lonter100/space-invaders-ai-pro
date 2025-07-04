# -*- coding: utf-8 -*-
"""
Moduł monitorowania wydajności Space Invaders AI Pro v3.0
==========================================================

Zaawansowany system monitorowania wydajności aplikacji AI.

Autor: lonter100
Wersja: 3.0
Licencja: MIT

Opis:
    Zawiera funkcje do monitorowania:
    - FPS i czas wykonania klatek
    - Użycie pamięci RAM i VRAM
    - Wydajność CPU i GPU
    - Statystyki AI i decyzji
    - Alerty o problemach z wydajnością
"""
import time
import psutil
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
import json
from pathlib import Path


class PerformanceMonitor:
    """
    Zaawansowany monitor wydajności dla aplikacji AI.
    
    Funkcje:
    - Monitorowanie FPS i czasu klatek
    - Śledzenie użycia zasobów systemowych
    - Analiza wydajności AI
    - Generowanie raportów i alertów
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Inicjalizacja monitora wydajności.
        
        Args:
            max_history: Maksymalna liczba pomiarów w historii
        """
        self.logger = logging.getLogger(__name__)
        self.max_history = max_history
        
        # Historia pomiarów
        self.frame_times = deque(maxlen=max_history)
        self.fps_history = deque(maxlen=max_history)
        self.memory_usage = deque(maxlen=max_history)
        self.cpu_usage = deque(maxlen=max_history)
        
        # Statystyki AI
        self.ai_decision_times = deque(maxlen=max_history)
        self.ai_accuracy = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=max_history)
        
        # Stan monitora
        self.is_running = False
        self.start_time = None
        self.frame_count = 0
        
        # Progi alertów
        self.fps_threshold = 30.0
        self.memory_threshold = 80.0  # %
        self.cpu_threshold = 90.0     # %
        
        # Thread do monitorowania zasobów
        self.monitor_thread = None
        
        self.logger.info("Monitor wydajności zainicjalizowany")
    
    def start(self) -> None:
        """Uruchom monitoring wydajności."""
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Uruchom thread do monitorowania zasobów
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Monitoring wydajności uruchomiony")
    
    def stop(self) -> None:
        """Zatrzymaj monitoring wydajności."""
        self.is_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        self.logger.info("Monitoring wydajności zatrzymany")
    
    def update(self, frame_time: float, ai_decision_time: Optional[float] = None,
               reward: Optional[float] = None, accuracy: Optional[float] = None) -> None:
        """
        Zaktualizuj statystyki wydajności.
        
        Args:
            frame_time: Czas wykonania klatki w sekundach
            ai_decision_time: Czas decyzji AI w sekundach
            reward: Nagroda AI
            accuracy: Dokładność AI (0.0-1.0)
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Aktualizuj historię
        self.frame_times.append(frame_time)
        
        if ai_decision_time is not None:
            self.ai_decision_times.append(ai_decision_time)
        
        if reward is not None:
            self.reward_history.append(reward)
        
        if accuracy is not None:
            self.ai_accuracy.append(accuracy)
        
        # Sprawdź alerty
        self._check_alerts()
    
    def _monitor_resources(self) -> None:
        """Monitoruj zasoby systemowe w osobnym thread."""
        while self.is_running:
            try:
                # Użycie pamięci
                memory_percent = psutil.virtual_memory().percent
                self.memory_usage.append(memory_percent)
                
                # Użycie CPU
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)
                
                # Oblicz FPS
                if len(self.frame_times) > 1:
                    recent_fps = 1.0 / np.mean(list(self.frame_times)[-10:])
                    self.fps_history.append(recent_fps)
                
                time.sleep(0.5)  # Aktualizuj co 0.5s
                
            except Exception as e:
                self.logger.error(f"Błąd monitorowania zasobów: {e}")
                time.sleep(1.0)
    
    def _check_alerts(self) -> None:
        """Sprawdź czy przekroczono progi alertów."""
        if len(self.fps_history) > 0:
            current_fps = self.fps_history[-1]
            if current_fps < self.fps_threshold:
                self.logger.warning(f"Niska wydajność FPS: {current_fps:.2f}")
        
        if len(self.memory_usage) > 0:
            current_memory = self.memory_usage[-1]
            if current_memory > self.memory_threshold:
                self.logger.warning(f"Wysokie użycie pamięci: {current_memory:.1f}%")
        
        if len(self.cpu_usage) > 0:
            current_cpu = self.cpu_usage[-1]
            if current_cpu > self.cpu_threshold:
                self.logger.warning(f"Wysokie użycie CPU: {current_cpu:.1f}%")
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Pobierz aktualne statystyki wydajności.
        
        Returns:
            Słownik ze statystykami
        """
        stats = {
            'frame_count': self.frame_count,
            'total_time': time.time() - self.start_time if self.start_time else 0.0,
        }
        
        if len(self.frame_times) > 0:
            stats.update({
                'avg_frame_time': np.mean(self.frame_times),
                'min_frame_time': np.min(self.frame_times),
                'max_frame_time': np.max(self.frame_times),
                'frame_time_std': np.std(self.frame_times),
            })
        
        if len(self.fps_history) > 0:
            stats.update({
                'current_fps': self.fps_history[-1],
                'avg_fps': np.mean(self.fps_history),
                'min_fps': np.min(self.fps_history),
                'max_fps': np.max(self.fps_history),
            })
        
        if len(self.memory_usage) > 0:
            stats.update({
                'current_memory': self.memory_usage[-1],
                'avg_memory': np.mean(self.memory_usage),
                'max_memory': np.max(self.memory_usage),
            })
        
        if len(self.cpu_usage) > 0:
            stats.update({
                'current_cpu': self.cpu_usage[-1],
                'avg_cpu': np.mean(self.cpu_usage),
                'max_cpu': np.max(self.cpu_usage),
            })
        
        if len(self.ai_decision_times) > 0:
            stats.update({
                'avg_ai_decision_time': np.mean(self.ai_decision_times),
                'max_ai_decision_time': np.max(self.ai_decision_times),
            })
        
        if len(self.reward_history) > 0:
            stats.update({
                'avg_reward': np.mean(self.reward_history),
                'total_reward': np.sum(self.reward_history),
                'min_reward': np.min(self.reward_history),
                'max_reward': np.max(self.reward_history),
            })
        
        if len(self.ai_accuracy) > 0:
            stats.update({
                'avg_accuracy': np.mean(self.ai_accuracy),
                'current_accuracy': self.ai_accuracy[-1],
            })
        
        return stats
    
    def generate_report(self, output_path: str = 'performance_report.json') -> None:
        """
        Wygeneruj raport wydajności.
        
        Args:
            output_path: Ścieżka do pliku raportu
        """
        try:
            stats = self.get_statistics()
            
            # Dodaj metadane
            report = {
                'timestamp': time.time(),
                'version': '3.0',
                'statistics': stats,
                'alerts': self._get_alerts(),
                'recommendations': self._get_recommendations(stats),
            }
            
            # Zapisz raport
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Raport wydajności zapisany: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Błąd podczas generowania raportu: {e}")
    
    def _get_alerts(self) -> List[str]:
        """Pobierz listę alertów wydajności."""
        alerts = []
        
        if len(self.fps_history) > 0 and self.fps_history[-1] < self.fps_threshold:
            alerts.append(f"Niska wydajność FPS: {self.fps_history[-1]:.2f}")
        
        if len(self.memory_usage) > 0 and self.memory_usage[-1] > self.memory_threshold:
            alerts.append(f"Wysokie użycie pamięci: {self.memory_usage[-1]:.1f}%")
        
        if len(self.cpu_usage) > 0 and self.cpu_usage[-1] > self.cpu_threshold:
            alerts.append(f"Wysokie użycie CPU: {self.cpu_usage[-1]:.1f}%")
        
        return alerts
    
    def _get_recommendations(self, stats: Dict[str, float]) -> List[str]:
        """Pobierz rekomendacje na podstawie statystyk."""
        recommendations = []
        
        if 'avg_fps' in stats and stats['avg_fps'] < 30:
            recommendations.append("Rozważ zmniejszenie rozdzielczości lub jakości grafiki")
        
        if 'avg_memory' in stats and stats['avg_memory'] > 80:
            recommendations.append("Zamknij inne aplikacje aby zwolnić pamięć")
        
        if 'avg_cpu' in stats and stats['avg_cpu'] > 90:
            recommendations.append("Sprawdź procesy w tle zużywające CPU")
        
        if 'avg_ai_decision_time' in stats and stats['avg_ai_decision_time'] > 0.1:
            recommendations.append("Rozważ optymalizację algorytmów AI")
        
        return recommendations
    
    def reset(self) -> None:
        """Zresetuj wszystkie statystyki."""
        self.frame_times.clear()
        self.fps_history.clear()
        self.memory_usage.clear()
        self.cpu_usage.clear()
        self.ai_decision_times.clear()
        self.ai_accuracy.clear()
        self.reward_history.clear()
        
        self.frame_count = 0
        self.start_time = time.time()
        
        self.logger.info("Statystyki wydajności zresetowane") 