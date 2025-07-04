# -*- coding: utf-8 -*-
"""
Moduł AI Utils
Autor: Twój Nick
Wersja: 2.0
Licencja: MIT
Opis: Narzędzia AI do detekcji, decyzji i obsługi agenta DQN.
"""
import cv2
import numpy as np
import pytesseract
from configs.config import PLAYER_COLOR_LOWER, PLAYER_COLOR_UPPER, ENEMY_COLOR_LOWER, ENEMY_COLOR_UPPER, SCORE_REGION, SCORE_THRESHOLD, TEMPLATE_MATCH_THRESHOLD, TEMPLATE_PLAYER_PATH, TEMPLATE_ENEMY_PATH
import torch
import torch.nn as nn
import torch.optim as optim
import random
from ai.replay_buffer import ReplayBuffer
import os
import time

logfile = open('log.txt', 'a')

ACTIONS = ['left', 'right', 'up', 'down', 'space']
MENU_ACTIONS = ['menu_up', 'menu_down', 'menu_left', 'menu_right', 'menu_select']

PLAYER_TEMPLATE_PATH = 'player_template.png'
ENEMY_TEMPLATE_PATH = 'enemy_template.png'

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv(o)
        return int(np.prod(o.size()))
    def forward(self, x):
        x = x.float() / 255.0
        conv_out = self.conv(x).reshape(x.size()[0], -1)
        return self.fc(conv_out)

class DQNAgent:
    def __init__(self, input_shape, n_actions, device='cuda', buffer_size=10000, batch_size=32, gamma=0.99, lr=1e-4):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = DQN(input_shape, n_actions).to(self.device)
        self.target_model = DQN(input_shape, n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learn_step = 0
        self.update_target_steps = 1000
        self.model_path = 'dqn_model.pth'
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict())
    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            state_v = torch.tensor(np.array([state]), device=self.device).permute(0,3,1,2)
            q_vals = self.model(state_v)
            _, action = torch.max(q_vals, dim=1)
            action = int(action.item())
        logfile.write(f'DQN: {ACTIONS[action]}\n')
        return ACTIONS[action]
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state, device=self.device).permute(0,3,1,2).float() / 255.0
        next_state = torch.tensor(next_state, device=self.device).permute(0,3,1,2).float() / 255.0
        action = torch.tensor(action, device=self.device).long()
        reward = torch.tensor(reward, device=self.device).float()
        done = torch.tensor(done, device=self.device).float()
        q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_state).max(1)[0]
            expected_q = reward + self.gamma * next_q * (1 - done)
        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step += 1
        if self.learn_step % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
    def save(self):
        torch.save(self.model.state_dict(), self.model_path)
        logfile.write(f'Model zapisany do pliku: {self.model_path}\n')

# Template matching (opcjonalnie)
def template_match(img, template_path, threshold):
    if not os.path.exists(template_path):
        return None
    try:
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        if template is None:
            return None
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            pt = (int(np.mean(loc[1])), int(np.mean(loc[0])))
            return pt
    except Exception as e:
        logfile.write(f'Błąd template matching: {e}\n')
    return None

def detect_object(img, template_path, threshold=0.8):
    if not os.path.exists(template_path):
        return None
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        return None
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        h, w = template.shape[:2]
        center = (max_loc[0] + w // 2, max_loc[1] + h // 2)
        return center
    return None

def detect_player(img):
    pt = detect_object(img, PLAYER_TEMPLATE_PATH, TEMPLATE_MATCH_THRESHOLD)
    if pt:
        return pt
    lower = np.array(PLAYER_COLOR_LOWER)
    upper = np.array(PLAYER_COLOR_UPPER)
    mask = cv2.inRange(img, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return (x + w // 2, y + h // 2)
    return None

def detect_enemies(img):
    enemies = []
    pt = detect_object(img, ENEMY_TEMPLATE_PATH, TEMPLATE_MATCH_THRESHOLD)
    if pt:
        # Dodaj bounding box wokół wykrytego szablonu (przykładowo 32x32 px)
        ex, ey = pt
        enemies.append((ex-16, ey-16, 32, 32))
    lower = np.array(ENEMY_COLOR_LOWER)
    upper = np.array(ENEMY_COLOR_UPPER)
    mask = cv2.inRange(img, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # pomijaj szum
            enemies.append((x, y, w, h))
    return enemies

def read_score(img):
    region = img[0:97, 0:185]  # Region punktacji wg @5.JPG
    if region.size == 0:
        return 0
    try:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, SCORE_THRESHOLD, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        text = text.strip()
        if not text:
            return 0
        score = int(''.join(filter(str.isdigit, text))) if any(c.isdigit() for c in text) else 0
    except Exception as e:
        score = 0
    return score

# Nowe: szybkie i bezpieczne OCR menu/gameover tylko na wycinku
from threading import Thread
import queue

def fast_ocr(img, timeout=1.5):
    q = queue.Queue()
    def ocr_worker():
        try:
            text = pytesseract.image_to_string(img, config='--psm 6').upper()
            q.put(text)
        except Exception as e:
            q.put('')
    t = Thread(target=ocr_worker)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return ''
    return q.get() if not q.empty() else ''

def detect_gameover(img):
    # OCR tylko na wycinku z napisem HALL OF FAME (prawa część ekranu)
    h, w = img.shape[:2]
    region = img[0:h, int(w*0.5):w]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    text = fast_ocr(gray)
    if 'HALL OF FAME' in text:
        return True
    return False

def detect_menu(img):
    # OCR tylko na wycinku z napisami menu (środek ekranu)
    h, w = img.shape[:2]
    region = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    text = fast_ocr(gray)
    menu_keywords = ['PLAY', 'SOUND', 'CONTROLS', 'QUIT']
    for word in menu_keywords:
        if word in text:
            return True
    return False

def detect_lives(img):
    # Region: 4 statki w rzędzie, poszerzony w lewo do krawędzi okna
    region = img[180:240, 0:180]  # y1:y2, x1:x2
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lives = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 10 < w < 40 and 10 < h < 40:
            lives += 1
    return lives

def detect_menu_play_selected(img):
    # Zakładamy, że czerwona kropka ma kolor zbliżony do (BGR) (0,0,255) i jest po lewej stronie napisu PLAY
    # Wycinamy region wokół napisu PLAY (np. środek ekranu, szerokość ok. 100x30 px)
    h, w = img.shape[:2]
    region = img[int(h*0.32):int(h*0.38), int(w*0.36):int(w*0.56)]
    # Szukamy czerwonych pikseli
    lower_red = np.array([0,0,180])
    upper_red = np.array([60,60,255])
    mask = cv2.inRange(region, lower_red, upper_red)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw > 2 and rh > 2:
            return True
    return False

# --- AI shooting cooldown (global for this module) ---
ai_shoot_cooldown = {'last_shot': 0, 'cooldown': 8}

def ai_decision(player_pos, enemies, dqn_agent=None, state=None, menu=False, img=None, frame_count=None):
    if menu:
        if img is not None and detect_menu_play_selected(img):
            return 'menu_select'
        else:
            return None
    if dqn_agent is not None and state is not None:
        action = dqn_agent.select_action(state)
        return action
    # Ulepszona logika: AI strzela, jeśli jakakolwiek część bounding boxa wroga jest na linii strzału.
    # Omija cały obszar wroga, nie tylko środek.
    if player_pos and enemies:
        px, py = player_pos
        for enemy in enemies:
            if len(enemy) == 4:
                ex, ey, ew, eh = enemy
                # Strzał: gracz w poziomym zakresie wroga i wróg poniżej gracza
                if ex <= px <= ex + ew and ey < py:
                    return 'space'
                # Omijanie: jeśli gracz jest blisko bounding boxa wroga, uciekaj w bok
                if abs(px - (ex + ew // 2)) < ew // 2 + 10 and abs(py - (ey + eh // 2)) < eh // 2 + 20:
                    if px < ex + ew // 2:
                        return 'left'
                    else:
                        return 'right'
            else:
                # fallback dla [(ex, ey), ...]
                ex, ey = enemy[:2]
                if abs(ex - px) < 15 and ey < py:
                    return 'space'
                if abs(ex - px) < 20 and abs(ey - py) < 30:
                    if px < ex:
                        return 'left'
                    else:
                        return 'right'
        # Jeśli nie ma zagrożenia ani celu, strzelaj lub idź do najbliższego wroga
        return 'space'
    return 'left'

def read_round(img):
    # Zakładamy, że pasek z napisem ROUND X jest w górnej części ekranu
    h, w = img.shape[:2]
    region = img[0:int(h*0.15), int(w*0.25):int(w*0.85)]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 7 -c tessedit_char_whitelist=0123456789ROUND ')
    text = text.strip().upper()
    if 'ROUND' in text:
        parts = text.split()
        for i, part in enumerate(parts):
            if part == 'ROUND' and i+1 < len(parts):
                try:
                    return int(parts[i+1])
                except Exception:
                    continue
    return 1  # domyślnie runda 1 jeśli nie wykryto 