import numpy as np
import random
import torch

ACTIONS = ['left', 'right', 'up', 'down', 'space']

class SimpleSpaceInvadersSim:
    def __init__(self, width=84, height=84, n_enemies=3):
        self.width = width
        self.height = height
        self.n_enemies = n_enemies
        self.reset()
        self.shoot_cooldown = 0  # cooldown na strzał

    def reset(self):
        self.player = [self.width // 2, self.height - 10]
        self.enemies = [[random.randint(10, self.width-10), random.randint(10, self.height//2)] for _ in range(self.n_enemies)]
        self.done = False
        self.steps = 0
        return self._get_state()

    def step(self, action):
        # Ruch gracza
        if action == 0:  # left
            self.player[0] = max(0, self.player[0] - 5)
        elif action == 1:  # right
            self.player[0] = min(self.width-1, self.player[0] + 5)
        elif action == 2:  # up
            self.player[1] = max(0, self.player[1] - 5)
        elif action == 3:  # down
            self.player[1] = min(self.height-1, self.player[1] + 5)
        # Ruch wrogów (prosto w dół)
        for e in self.enemies:
            e[1] += 2
        # Strzał
        reward = 0
        shot_fired = False
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if action == 4 and self.shoot_cooldown == 0:
            shot_fired = True
            self.shoot_cooldown = 8  # gracz może strzelać co 8 klatek
            hit = False
            for e in self.enemies:
                if abs(e[0] - self.player[0]) < 8 and 0 < self.player[1] - e[1] < 20:
                    reward += 120  # większa nagroda za celny strzał
                    e[1] = -100  # usunięty wróg
                    hit = True
            if not hit:
                reward -= 10  # kara za niecelny strzał
        # Kara za spamowanie strzałów
        if action == 4 and not shot_fired:
            reward -= 2
        # Kara za kolizję
        for e in self.enemies:
            dist = np.linalg.norm(np.array(e) - np.array(self.player))
            if dist < 8:
                reward -= 700  # większa kara za kolizję
                self.done = True
            elif dist < 16:
                reward -= 20  # kara za bardzo bliskie podejście
            elif dist < 32:
                reward -= 5   # lekka kara za zbliżenie się
            elif dist < 48 and prev_distance < 48 and dist > prev_distance:
                reward += 10  # nagroda za oddalanie się od wroga
        # Kara za wyjście poza ekran
        if self.player[0] < 0 or self.player[0] >= self.width or self.player[1] < 0 or self.player[1] >= self.height:
            reward -= 100
            self.done = True
        # Nagroda za przetrwanie
        reward += 1
        # Usuwanie zestrzelonych wrogów
        self.enemies = [e for e in self.enemies if e[1] >= 0]
        # Koniec epizodu jeśli brak wrogów
        if not self.enemies:
            self.done = True
            reward += 500
        self.steps += 1
        if self.steps > 200:
            self.done = True
        return self._get_state(), reward, self.done

    def _get_state(self):
        # Zwraca uproszczony obraz (tablica 84x84x3 z gracza i wrogami)
        state = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Gracz: niebieski
        cvx, cvy = self.player
        state[max(0, cvy-2):min(self.height, cvy+3), max(0, cvx-2):min(self.width, cvx+3), 2] = 255
        # Wrogowie: czerwoni
        for ex, ey in self.enemies:
            if 0 <= ex < self.width and 0 <= ey < self.height:
                state[max(0, ey-2):min(self.height, ey+3), max(0, ex-2):min(self.width, ex+3), 0] = 255
        return state

class AdvancedSpaceInvadersSim:
    def __init__(self, width=84, height=84, n_rows=5, n_cols=8, n_lives=3, enemy_speed=1, bullet_speed=3, enemy_bullet_speed=2, round_num=1, use_gpu=False):
        self.width = width
        self.height = height
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_lives = n_lives
        self.enemy_speed = enemy_speed
        self.bullet_speed = bullet_speed
        self.enemy_bullet_speed = enemy_bullet_speed
        self.round_num = round_num
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.reset()
        self.shoot_cooldown = 0
        self.max_player_bullets = 2  # gracz może mieć max 2 pociski naraz

    def reset(self):
        # Gracz
        self.player = [self.width // 2, self.height - 8]
        self.lives = self.n_lives
        self.score = 0
        # Wrogowie w formacji
        self.enemies = []
        spacing_x = self.width // (self.n_cols + 1)
        spacing_y = 6
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                x = spacing_x * (col + 1)
                y = 10 + row * spacing_y
                self.enemies.append({'pos': [x, y], 'alive': True, 'row': row, 'col': col})
        self.enemy_dir = 1  # 1: prawo, -1: lewo
        self.enemy_move_down = False
        self.enemy_move_counter = 0
        self.enemy_move_interval = max(2, 10 - self.round_num)
        # Pociski
        self.player_bullets = []
        self.enemy_bullets = []
        self.done = False
        self.round_done = False
        self.steps = 0
        self.events = []
        return self._get_state()

    def step(self, action):
        self.events = []
        reward = 0
        # Ruch gracza
        if action == 0:  # left
            self.player[0] = max(0, self.player[0] - 4)
        elif action == 1:  # right
            self.player[0] = min(self.width-1, self.player[0] + 4)
        elif action == 2:  # up
            self.player[1] = max(self.height//2, self.player[1] - 2)
        elif action == 3:  # down
            self.player[1] = min(self.height-1, self.player[1] + 2)
        # Strzał gracza
        shot_fired = False
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if action == 4 and len(self.player_bullets) < self.max_player_bullets and self.shoot_cooldown == 0:
            self.player_bullets.append({'pos': [self.player[0], self.player[1]-2]})
            self.shoot_cooldown = 6  # gracz może strzelać co 6 klatek
            shot_fired = True
        # Kara za spamowanie strzałów
        if action == 4 and not shot_fired:
            reward -= 3
        # Ruch pocisków gracza
        for b in self.player_bullets:
            b['pos'][1] -= self.bullet_speed
        self.player_bullets = [b for b in self.player_bullets if b['pos'][1] > 0]
        # Ruch pocisków wrogów
        for b in self.enemy_bullets:
            b['pos'][1] += self.enemy_bullet_speed
        self.enemy_bullets = [b for b in self.enemy_bullets if b['pos'][1] < self.height]
        # Ruch wrogów (formacja)
        self.enemy_move_counter += 1
        if self.enemy_move_counter >= self.enemy_move_interval:
            self.enemy_move_counter = 0
            edge_hit = False
            for e in self.enemies:
                if not e['alive']:
                    continue
                e['pos'][0] += self.enemy_dir * self.enemy_speed
                if e['pos'][0] < 5 or e['pos'][0] > self.width-5:
                    edge_hit = True
            if edge_hit:
                self.enemy_dir *= -1
                for e in self.enemies:
                    if e['alive']:
                        e['pos'][1] += 4
        # Strzały wrogów (losowo z dolnego rzędu)
        if random.random() < 0.08:
            bottom_enemies = {}
            for e in self.enemies:
                if not e['alive']:
                    continue
                col = e['col']
                if col not in bottom_enemies or e['pos'][1] > bottom_enemies[col]['pos'][1]:
                    bottom_enemies[col] = e
            if bottom_enemies:
                shooter = random.choice(list(bottom_enemies.values()))
                self.enemy_bullets.append({'pos': [shooter['pos'][0], shooter['pos'][1]+2]})
        # Kolizje pocisków gracza z wrogami
        hit_any = False
        for b in self.player_bullets:
            for e in self.enemies:
                if e['alive'] and abs(b['pos'][0] - e['pos'][0]) < 4 and abs(b['pos'][1] - e['pos'][1]) < 4:
                    e['alive'] = False
                    reward += 150  # większa nagroda za celny strzał
                    self.score += 150
                    self.events.append('enemy_killed')
                    hit_any = True
        if shot_fired and not hit_any:
            reward -= 12  # kara za niecelny strzał
            self.events.append('missed_shot')
        self.player_bullets = [b for b in self.player_bullets if not any(e['alive']==False and abs(b['pos'][0]-e['pos'][0])<4 and abs(b['pos'][1]-e['pos'][1])<4 for e in self.enemies)]
        # Kolizje pocisków wrogów z graczem
        for b in self.enemy_bullets:
            if abs(b['pos'][0] - self.player[0]) < 4 and abs(b['pos'][1] - self.player[1]) < 4:
                self.lives -= 1
                reward -= 200
                self.events.append('player_hit')
        self.enemy_bullets = [b for b in self.enemy_bullets if not (abs(b['pos'][0] - self.player[0]) < 4 and abs(b['pos'][1] - self.player[1]) < 4)]
        # Kolizje wrogów z graczem (przegrana)
        for e in self.enemies:
            if e['alive'] and abs(e['pos'][0] - self.player[0]) < 6 and abs(e['pos'][1] - self.player[1]) < 6:
                self.lives = 0
                reward -= 500
                self.events.append('enemy_collision')
        # Usuwanie martwych wrogów
        alive_enemies = [e for e in self.enemies if e['alive']]
        # --- Nowa logika: nagroda za unikanie wrogów ---
        min_dist = min([np.linalg.norm(np.array(self.player) - np.array(e['pos'])) for e in alive_enemies] or [self.height])
        if min_dist < 12:
            reward -= 15  # większa kara za zbliżenie się do wroga
            self.events.append('too_close_to_enemy')
        elif min_dist < 20:
            reward += 4  # umiarkowana nagroda za trzymanie dystansu
            self.events.append('safe_distance')
        elif min_dist >= 20:
            reward += 8  # duża nagroda za bardzo bezpieczną pozycję
            self.events.append('very_safe_distance')
        # Dodatkowa nagroda za oddalanie się od najbliższego wroga
        if hasattr(self, 'prev_min_dist'):
            if min_dist > self.prev_min_dist + 4:
                reward += 10
                self.events.append('moved_away_from_enemy')
        self.prev_min_dist = min_dist
        # --- Koniec nowej logiki ---
        # Koniec rundy
        if not alive_enemies:
            self.round_done = True
            reward += 500
            self.events.append('round_cleared')
        # Przegrana
        if self.lives <= 0:
            self.done = True
            reward -= 1000
            self.events.append('game_over')
        # Nagroda za przetrwanie
        reward += 1
        self.steps += 1
        if self.steps > 1000:
            self.done = True
        return self._get_state(), reward, self.done, self.events

    def _get_state(self):
        # Zwraca obraz RGB (tablica height x width x 3) z gracza, wrogami, pociskami, liczbą żyć, wynikiem
        state = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Gracz: niebieski
        cvx, cvy = self.player
        state[max(0, cvy-3):min(self.height, cvy+4), max(0, cvx-3):min(self.width, cvx+4), 2] = 255
        # Wrogowie: czerwoni
        for e in self.enemies:
            if e['alive']:
                ex, ey = e['pos']
                state[max(0, ey-3):min(self.height, ey+4), max(0, ex-3):min(self.width, ex+4), 0] = 255
        # Pociski gracza: zielone
        for b in self.player_bullets:
            bx, by = b['pos']
            state[max(0, by-1):min(self.height, by+2), max(0, bx-1):min(self.width, bx+2), 1] = 255
        # Pociski wrogów: żółte
        for b in self.enemy_bullets:
            bx, by = b['pos']
            state[max(0, by-1):min(self.height, by+2), max(0, bx-1):min(self.width, bx+2), 0:2] = 255
        # Liczba żyć i wynik (opcjonalnie można zakodować w stanie lub zwracać osobno)
        if self.use_gpu:
            return torch.from_numpy(state).float().permute(2,0,1).cuda(non_blocking=True) / 255.0
        return state

def generate_simulated_batch(sim, batch_size=64):
    batch = []
    for _ in range(batch_size):
        state = sim.reset()
        done = False
        while not done:
            action = random.randint(0, 4)
            next_state, reward, done = sim.step(action)
            batch.append((state, action, reward, next_state, done))
            state = next_state
            if len(batch) >= batch_size:
                break
        if len(batch) >= batch_size:
            break
    return batch 