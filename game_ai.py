import time
import cv2
import torch
import os
import yaml
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional
from ai.reward import RewardConfig, calc_reward
from core.logger import setup_logger
from ai.ai_utils import detect_player, detect_enemies, read_score, ai_decision, DQNAgent, detect_gameover, detect_menu, detect_lives, read_round
from core.screen_utils import find_game_region, grab_screen, save_lives_region
from core.controller import perform_action
from configs.config import TEMPLATE_PLAYER_PATH, TEMPLATE_ENEMY_PATH, SCORE_REGION
import pytesseract
from ai.agent import get_best_agent, pretrain_agent
from ai.prediction import EnemyTracker, ShotAnalyzer

class GameAI:
    def __init__(self, config_path='configs/config.yaml'):
        self.logger = setup_logger()
        self.reward_cfg = RewardConfig(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'Używane urządzenie: {self.device}')
        if not os.path.exists(TEMPLATE_PLAYER_PATH):
            self.logger.warning('Brak szablonu gracza (player_template.png) – wykrywanie tylko po kolorze!')
        if not os.path.exists(TEMPLATE_ENEMY_PATH):
            self.logger.warning('Brak szablonu wroga (enemy_template.png) – wykrywanie tylko po kolorze!')
        
        # Initialize game region and screen properties
        self.game_region = find_game_region()
        self.screen_width = self.game_region['width']
        self.screen_height = self.game_region['height']
        
        # Initialize tracking and prediction systems
        self.enemy_tracker = EnemyTracker()
        self.shot_analyzer = ShotAnalyzer(self.screen_width, self.screen_height)
        self.enemy_positions: Dict[int, Tuple[float, float, float, float]] = {}
        self.predicted_enemy_positions: Dict[int, List[Tuple[float, float]]] = {}
        self.last_shot_positions: List[Tuple[float, float]] = []
        
        # Game state
        img = grab_screen(self.game_region)
        self.input_shape = (3, 84, 84)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config.get('agent', {})
        self.agent = get_best_agent(self.input_shape, n_actions=5, device=self.device, config=self.config)
        
        # Initialize state tracking
        self.prev_score = 0
        self.prev_state = self.preprocess(img)
        self.prev_enemies = detect_enemies(img)
        self.prev_lives = detect_lives(img)
        self.frame_count = 0
        self.score = self.prev_score
        self.lives = self.prev_lives
        self.done = False
        self.menu = False
        self.last_state_hash = None
        self.prev_round = 1
        self.round = 1
        self.idle_counter = 0
        self.last_shot_frame = 0
        self.last_shot_hit = False
        self.last_score = 0
        self.edge_threshold = 30  # px from edge
        self.prev_action = None
        self.SAFE_DISTANCE = 40  # px
        self.VERY_SAFE_DISTANCE = 80  # px
        self.total_reward = 0.0
        self.last_logged_round = 1

    def _is_at_edge(self, player_pos: Tuple[float, float]) -> bool:
        """Check if player is near the edge of the screen"""
        x, y = player_pos
        return (x < self.edge_threshold or 
                x > self.screen_width - self.edge_threshold or
                y < self.edge_threshold or 
                y > self.screen_height - self.edge_threshold)
    
    def _is_too_close_to_enemy(self, player_pos: Tuple[float, float], 
                             enemies: List[Tuple[float, float]]) -> bool:
        """Check if player is too close to any enemy"""
        if not player_pos or not enemies:
            return False
        
        player_x, player_y = player_pos
        for enemy in enemies:
            if len(enemy) >= 2:  # Ensure enemy has at least x,y coordinates
                dist = np.sqrt((enemy[0] - player_x)**2 + (enemy[1] - player_y)**2)
                if dist < self.SAFE_DISTANCE:
                    return True
        return False
    
    def _is_formation_broken(self, prev_enemies: List[Tuple[float, float]], 
                           current_enemies: List[Tuple[float, float]]) -> bool:
        """Check if enemy formation has been broken"""
        if len(prev_enemies) < 3 or len(current_enemies) < 3:
            return False
            
        prev_formation = (self.detect_row_formation(prev_enemies) or
                         self.detect_column_formation(prev_enemies) or
                         self.detect_cluster_formation(prev_enemies))
                         
        current_formation = (self.detect_row_formation(current_enemies) or
                           self.detect_column_formation(current_enemies) or
                           self.detect_cluster_formation(current_enemies))
                           
        return prev_formation and not current_formation
    
    def preprocess(self, img):
        """Preprocess game screen for the neural network"""
        # Resize and normalize
        img = cv2.resize(img, (84, 84))
        
        # Optional: Convert to grayscale to reduce input dimensions
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = np.expand_dims(img, axis=-1)
        
        return img  # (84, 84, 3) RGB or (84, 84, 1) for grayscale

    def detect_row_formation(self, enemies, tolerance=10):
        """Detect if enemies are aligned in a row within given y-tolerance"""
        if len(enemies) < 3:
            return False
        # Convert to numpy array if needed
        enemies_array = np.array([(x, y) for (x, y) in enemies])
        ys = enemies_array[:, 1]
        return np.std(ys) < tolerance

    def detect_column_formation(self, enemies, tolerance=10):
        """Detect if enemies are aligned in a column within given x-tolerance"""
        if len(enemies) < 3:
            return False
        # Convert to numpy array if needed
        enemies_array = np.array([(x, y) for (x, y) in enemies])
        xs = enemies_array[:, 0]
        return np.std(xs) < tolerance

    def detect_cluster_formation(self, enemies, max_distance=40):
        """Detect if enemies are clustered together within max_distance"""
        if len(enemies) < 3:
            return False
        enemies_array = np.array([(x, y) for (x, y) in enemies])
        dists = np.linalg.norm(enemies_array - enemies_array.mean(axis=0), axis=1)
        return np.max(dists) < max_distance

    def _update_enemy_tracking(self, current_enemies: List[Tuple[float, float]]) -> None:
        """Update enemy tracking and prediction"""
        # Update enemy positions and velocities
        self.enemy_positions = self.enemy_tracker.update(current_enemies)
        
        # Predict future positions (next 30 frames)
        self.predicted_enemy_positions = self.enemy_tracker.predict_future_positions(steps=30)
    
    def _analyze_shot(self, shot_pos: Tuple[float, float], 
                     player_pos: Tuple[float, float], 
                     current_enemies: List[Tuple[float, float]]) -> None:
        """Analyze the result of a shot"""
        if not hasattr(self, 'last_shot_pos'):
            self.last_shot_pos = shot_pos
            self.last_shot_frame = self.frame_count
            return
        
        # Check if any enemies were hit
        enemy_hit = False
        for enemy_id, (x, y, vx, vy) in self.enemy_positions.items():
            # Simple proximity check (could be enhanced with actual bullet tracking)
            if abs(x - self.last_shot_pos[0]) < 20 and abs(y - self.last_shot_pos[1]) < 20:
                enemy_hit = True
                break
        
        # Update shot analyzer
        self.shot_analyzer.add_shot(shot_pos, enemy_hit)
        self.last_shot_hit = enemy_hit
        self.last_shot_pos = shot_pos
        self.last_shot_frame = self.frame_count
    
    def get_optimal_shot_position(self) -> Tuple[float, float]:
        """Get the optimal position to shoot based on enemy prediction"""
        player_pos = detect_player(grab_screen(self.game_region))
        if not player_pos:
            return None
            
        # Get current enemy positions with velocities
        enemies = list(self.enemy_positions.values())
        
        # Use shot analyzer to find optimal shot
        return self.shot_analyzer.get_optimal_shot_position(player_pos, enemies)
    
    def get_danger_zones(self) -> List[Tuple[float, float, float]]:
        """Get dangerous areas to avoid"""
        return self.shot_analyzer.get_danger_zones()
    
    def get_events(self, prev_score, score, prev_enemies, enemies, prev_lives, lives, done, player_pos=None, img=None):
        # Calculate basic events
        enemy_removed = len(enemies) < len(prev_enemies)
        enemy_delta = max(0, len(prev_enemies) - len(enemies))
        life_lost = lives < prev_lives
        life_delta = max(0, prev_lives - lives)
        
        # Enhanced events based on prediction
        predicted_danger = False
        if player_pos and self.predicted_enemy_positions:
            for enemy_id, positions in self.predicted_enemy_positions.items():
                for x, y in positions[:10]:  # Check next 10 frames
                    if abs(x - player_pos[0]) < 30 and abs(y - player_pos[1]) < 30:
                        predicted_danger = True
                        break
                if predicted_danger:
                    break
        
        events = {
            'done': done,
            'score_increased': score > prev_score,
            'score_delta': max(0, score - prev_score),
            'enemy_removed': enemy_removed,
            'enemy_delta': enemy_delta,
            'life_lost': life_lost,
            'life_delta': life_delta,
            'survived': not done,
            'avoided_enemy': predicted_danger and not life_lost,
            'fast_kill': enemy_removed and (self.frame_count - self.last_shot_frame) < 30,
            'idle': self.idle_counter > 180,  # 3 seconds of no action
            'at_edge': self._is_at_edge(player_pos) if player_pos else False,
            'round_advanced': self.round > self.prev_round,
            'round_delta': max(0, self.round - self.prev_round),
            'cleared_wave': len(enemies) == 0 and len(prev_enemies) > 0,
            'missed_shots': 1 if not self.last_shot_hit and (self.frame_count - self.last_shot_frame) < 10 else 0,
            'formation_broken': self._is_formation_broken(prev_enemies, enemies),
            'row_broken': self.detect_row_formation(prev_enemies) and not self.detect_row_formation(enemies),
            'column_broken': self.detect_column_formation(prev_enemies) and not self.detect_column_formation(enemies),
            'cluster_broken': self.detect_cluster_formation(prev_enemies) and not self.detect_cluster_formation(enemies),
            'too_close_to_enemy': self._is_too_close_to_enemy(player_pos, enemies) if player_pos else False,
            'safe_distance': False,
            'very_safe_distance': False,
        }
        # Nagroda za przetrwanie co 100 klatek
        if self.frame_count % 100 == 0:
            events['survived'] = True
        # Kara za bezczynność (np. nie strzelał przez 50 klatek)
        if self.frame_count - self.last_shot_frame > 50:
            events['idle'] = True
        # Kara za krawędź ekranu
        if player_pos and img is not None:
            h, w = img.shape[:2]
            if player_pos[0] < self.edge_threshold or player_pos[0] > w - self.edge_threshold:
                events['at_edge'] = True
        # Nagroda za szybkie zestrzelenie
        if events['enemy_removed'] and (self.frame_count - self.last_enemy_kill_frame) < 10:
            events['fast_kill'] = True
        if events['enemy_removed']:
            self.last_enemy_kill_frame = self.frame_count
        # Nagroda za wyczyszczenie planszy
        if len(enemies) == 0 and len(prev_enemies) > 0:
            events['cleared_wave'] = True
        # Kara za strzały bez trafienia (np. 5 strzałów bez wzrostu score)
        if score == self.last_score and self.prev_action == 4:  # 4 = 'space'
            self.idle_counter += 1
        else:
            self.idle_counter = 0
        if self.idle_counter >= 5:
            events['missed_shots'] = self.idle_counter
        self.last_score = score
        # Wykrywanie rozbicia formacji
        if self.detect_row_formation(prev_enemies) and not self.detect_row_formation(enemies):
            events['row_broken'] = True
            events['formation_broken'] = True
        if self.detect_column_formation(prev_enemies) and not self.detect_column_formation(enemies):
            events['column_broken'] = True
            events['formation_broken'] = True
        if self.detect_cluster_formation(prev_enemies) and not self.detect_cluster_formation(enemies):
            events['cluster_broken'] = True
            events['formation_broken'] = True
        # Zaawansowana logika stref bezpieczeństwa
        if player_pos and enemies:
            dists = [np.linalg.norm(np.array(player_pos) - np.array(e)) for e in enemies]
            min_dist = min(dists) if dists else 999
            if min_dist < self.SAFE_DISTANCE:
                events['too_close_to_enemy'] = True
            elif self.SAFE_DISTANCE <= min_dist < self.VERY_SAFE_DISTANCE:
                events['safe_distance'] = True
            elif min_dist >= self.VERY_SAFE_DISTANCE:
                events['very_safe_distance'] = True
            if min_dist > self.SAFE_DISTANCE * 1.5:
                events['avoided_enemy'] = True
        return events

    def detect_hall_of_fame(self, img):
        h, w = img.shape[:2]
        region = img[0:int(h*0.2), 0:w]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 7').upper()
        return 'HALL OF FAME' in text

    def run(self):
        self.logger.info('AI uruchomione. Przerwij Ctrl+C.')
        while True:
            try:
                img = grab_screen(self.game_region)
                self.frame_count += 1
                state_hash = hash(img.tobytes())
                # Odświeżaj stan co każdą klatkę (nie co 10)
                # if self.frame_count % 10 == 0 or state_hash != self.last_state_hash:
                self.score = read_score(img)
                self.lives = detect_lives(img)
                self.done = detect_gameover(img) or (self.lives == 0)
                self.menu = detect_menu(img)
                self.round = read_round(img)
                self.last_state_hash = state_hash
                if self.detect_hall_of_fame(img):
                    self.logger.info('Wykryto ekran HALL OF FAME – AI naciska P')
                    import pyautogui
                    pyautogui.keyDown('p')
                    time.sleep(0.01)
                    pyautogui.keyUp('p')
                    time.sleep(1)
                    continue
                if self.menu:
                    self.logger.info('Wykryto menu gry – AI zawsze wybiera: P + ENTER')
                    perform_action('menu_select')
                    self.total_reward = 0.0  # Reset sumy nagród po wejściu do menu/po naciśnięciu play
                    time.sleep(1)
                    continue
                player_pos = detect_player(img)
                enemies = detect_enemies(img)
                state = self.preprocess(img)
                action_str = ai_decision(player_pos, enemies, dqn_agent=self.agent, state=state, menu=self.menu)
                perform_action(action_str)
                self.prev_action = action_str
                events = self.get_events(self.prev_score, self.score, self.prev_enemies, enemies, self.prev_lives, self.lives, self.done, player_pos, img)
                reward = calc_reward(self.prev_score, self.score, events, self.reward_cfg)
                self.total_reward += reward
                # Reset sumy nagród na początku nowej rundy
                if events.get('round_advanced') or self.round != self.last_logged_round:
                    self.logger.info(f'--- KONIEC RUNDY {self.prev_round} --- SUMA NAGRÓD: {self.total_reward:.2f}')
                    self.total_reward = 0.0
                    self.last_logged_round = self.round
                self.logger.info(f'Frame: {self.frame_count}, Score: {self.score}, Lives: {self.lives}, Action: {action_str}, Reward: {reward}, TotalReward: {self.total_reward:.2f}, Done: {self.done}')
                self.agent.remember(self.prev_state, action_str, reward, state, self.done)
                self.agent.update()
                self.prev_state = state
                self.prev_score = self.score
                self.prev_enemies = enemies
                self.prev_lives = self.lives
                self.prev_round = self.round
                overlay = img.copy()
                if player_pos:
                    cv2.circle(overlay, player_pos, 15, (255, 0, 0), 2)
                for e in enemies:
                    cv2.rectangle(overlay, (e[0]-10, e[1]-10), (e[0]+10, e[1]+10), (0, 255, 0), 2)
                cv2.putText(overlay, f'Score: {self.score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(overlay, f'Lives: {self.lives}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(overlay, f'Device: {self.device}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.imshow('AI podglad', overlay)
                try:
                    cv2.moveWindow('AI podglad', self.game_region['left'], self.game_region['top'])
                    cv2.resizeWindow('AI podglad', self.game_region['width'], self.game_region['height'])
                except Exception as e:
                    self.logger.warning(f'Nie można ustawić pozycji/rozmiaru okna podglądu: {e}')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # time.sleep(0.01)  # Usuwam lub zmieniam na 0, by zminimalizować opóźnienie
            except Exception as e:
                self.logger.error(f'Błąd główny: {e}')
        self.agent.save()
        cv2.destroyAllWindows() 