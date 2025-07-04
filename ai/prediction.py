import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
import cv2

class KalmanFilter:
    def __init__(self, dt=1.0, process_noise=1e-2, measurement_noise=1e-1):
        # State: [x, y, vx, vy]
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Initial state uncertainty
        self.P = np.eye(4)
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # State vector
        self.x = np.zeros((4, 1))
        self.initialized = False
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        if not self.initialized:
            self.x[:2] = np.array(measurement).reshape(2, 1)
            self.initialized = True
            return measurement
        
        # Prediction
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        y = np.array(measurement).reshape(2, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return (float(self.x[0, 0]), float(self.x[1, 0]))
    
    def predict(self, steps: int = 1) -> Tuple[float, float]:
        if not self.initialized:
            raise ValueError("Filter not initialized with measurements")
        
        x_pred = np.linalg.matrix_power(self.F, steps) @ self.x
        return (float(x_pred[0, 0]), float(x_pred[1, 0]))

class EnemyTracker:
    def __init__(self, max_history=10):
        self.enemy_filters: Dict[int, KalmanFilter] = {}
        self.enemy_history: Dict[int, deque] = {}
        self.max_history = max_history
        self.next_id = 0
    
    def update(self, current_enemies: List[Tuple[float, float]]) -> Dict[int, Tuple[float, float, float, float]]:
        """
        Update enemy positions and return a dictionary of enemy_id to (x, y, vx, vy)
        """
        if not current_enemies:
            return {}
        
        # Convert to numpy for efficient distance calculations
        current_positions = np.array(current_enemies)
        
        # If no existing filters, initialize them
        if not self.enemy_filters:
            for pos in current_positions:
                self._add_enemy(pos)
            return self._get_enemy_states()
        
        # Match current detections with existing filters using Hungarian algorithm
        cost_matrix = self._create_cost_matrix(current_positions)
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Update matched enemies
        matched = set()
        for row, col in zip(row_ind, col_ind):
            if row < len(self.enemy_filters) and col < len(current_positions):
                enemy_id = list(self.enemy_filters.keys())[row]
                self.enemy_filters[enemy_id].update(current_positions[col])
                self._update_enemy_history(enemy_id, current_positions[col])
                matched.add(enemy_id)
        
        # Remove lost enemies
        lost_enemies = set(self.enemy_filters.keys()) - matched
        for enemy_id in lost_enemies:
            if enemy_id in self.enemy_filters:
                del self.enemy_filters[enemy_id]
            if enemy_id in self.enemy_history:
                del self.enemy_history[enemy_id]
        
        # Add new enemies
        for i, pos in enumerate(current_positions):
            if i not in col_ind:
                self._add_enemy(pos)
        
        return self._get_enemy_states()
    
    def _add_enemy(self, position: Tuple[float, float]) -> None:
        enemy_id = self.next_id
        self.next_id += 1
        self.enemy_filters[enemy_id] = KalmanFilter()
        self.enemy_filters[enemy_id].update(position)
        self.enemy_history[enemy_id] = deque(maxlen=self.max_history)
        self._update_enemy_history(enemy_id, position)
    
    def _update_enemy_history(self, enemy_id: int, position: Tuple[float, float]) -> None:
        self.enemy_history[enemy_id].append(position)
    
    def _create_cost_matrix(self, current_positions: np.ndarray) -> np.ndarray:
        """Create cost matrix for Hungarian algorithm"""
        n = len(self.enemy_filters)
        m = len(current_positions)
        cost_matrix = np.zeros((max(n, m), max(n, m)))
        
        # Fill with large default cost
        cost_matrix.fill(1000)
        
        # Calculate costs between existing and current positions
        for i, (enemy_id, kf) in enumerate(self.enemy_filters.items()):
            for j, pos in enumerate(current_positions):
                # Predict next position based on current state
                predicted = kf.predict()
                # Calculate Euclidean distance between predicted and actual position
                cost = np.linalg.norm(np.array(predicted[:2]) - pos)
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _get_enemy_states(self) -> Dict[int, Tuple[float, float, float, float]]:
        """Return dictionary of enemy_id to (x, y, vx, vy)"""
        states = {}
        for enemy_id, kf in self.enemy_filters.items():
            x, y = kf.x[0, 0], kf.x[1, 0]
            vx, vy = kf.x[2, 0], kf.x[3, 0]
            states[enemy_id] = (x, y, vx, vy)
        return states
    
    def predict_future_positions(self, steps: int = 30) -> Dict[int, List[Tuple[float, float]]]:
        """Predict future positions of all tracked enemies"""
        predictions = {}
        for enemy_id, kf in self.enemy_filters.items():
            future_positions = []
            for step in range(1, steps + 1):
                x, y = kf.predict(step)
                future_positions.append((x, y))
            predictions[enemy_id] = future_positions
        return predictions

class ShotAnalyzer:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.shot_trajectories: List[List[Tuple[float, float]]] = []
        self.successful_shots: List[Tuple[float, float]] = []
        self.failed_shots: List[Tuple[float, float]] = []
    
    def add_shot(self, start_pos: Tuple[float, float], hit: bool, end_pos: Optional[Tuple[float, float]] = None):
        """Add a shot to the analyzer"""
        if hit:
            self.successful_shots.append(start_pos)
        else:
            self.failed_shots.append(start_pos)
    
    def get_optimal_shot_position(self, player_pos: Tuple[float, float], enemy_positions: List[Tuple[float, float, float, float]]) -> Tuple[float, float]:
        """
        Calculate the optimal position to shoot from based on enemy positions and movement
        Returns: (target_x, target_y)
        """
        if not enemy_positions:
            return player_pos[0], player_pos[1]  # Default to current position if no enemies
        
        # Convert to numpy for vectorized operations
        enemies = np.array([(x, y, vx, vy) for x, y, vx, vy in enemy_positions])
        
        # Calculate distances to player
        player_pos_np = np.array(player_pos)
        enemy_pos = enemies[:, :2]
        dists = np.linalg.norm(enemy_pos - player_pos_np, axis=1)
        
        # Find closest enemy
        closest_idx = np.argmin(dists)
        target_x, target_y, vx, vy = enemies[closest_idx]
        
        # Predict enemy position based on velocity (simple linear prediction)
        time_to_reach = dists[closest_idx] / 10.0  # Assuming constant bullet speed
        predicted_x = target_x + vx * time_to_reach
        predicted_y = target_y + vy * time_to_reach
        
        # Keep within screen bounds
        predicted_x = np.clip(predicted_x, 0, self.screen_width)
        predicted_y = np.clip(predicted_y, 0, self.screen_height)
        
        return predicted_x, predicted_y
    
    def get_danger_zones(self, radius: float = 50) -> List[Tuple[float, float, float]]:
        """
        Identify dangerous areas where enemies frequently hit the player
        Returns: List of (x, y, danger_level)
        """
        # Simple implementation: cluster successful shots
        if not self.successful_shots:
            return []
        
        from sklearn.cluster import DBSCAN
        
        shots = np.array(self.successful_shots)
        if len(shots) < 2:
            return []
        
        # Cluster successful shots
        clustering = DBSCAN(eps=radius, min_samples=2).fit(shots)
        
        danger_zones = []
        for label in set(clustering.labels_):
            if label == -1:  # Noise
                continue
            cluster = shots[clustering.labels_ == label]
            center = np.mean(cluster, axis=0)
            danger_level = len(cluster) / len(shots)
            danger_zones.append((center[0], center[1], danger_level))
        
        return danger_zones
