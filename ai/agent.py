import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from ai.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import os
import time
from ai.simulator import SimpleSpaceInvadersSim, generate_simulated_batch

# 1. Klasyczna sieć konwolucyjna (CNN)
class CNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv(o)
        return int(np.prod(o.size()))
    def forward(self, x):
        x = x.float() / 255.0
        conv_out = self.conv(x).reshape(x.size()[0], -1)
        return self.fc(conv_out)

# 2. Sieć konwolucyjno-rekurencyjna (CNN + LSTM)
class CNNRNN(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size=128):
        super(CNNRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.lstm = nn.LSTM(conv_out_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_actions)
    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv(o)
        return int(np.prod(o.size()))
    def forward(self, x, hidden=None):
        # x: (batch, seq, C, H, W)
        batch, seq, C, H, W = x.size()
        x = x.view(batch * seq, C, H, W)
        conv_out = self.conv(x).reshape(batch, seq, -1)
        if hidden is None:
            lstm_out, hidden = self.lstm(conv_out)
        else:
            lstm_out, hidden = self.lstm(conv_out, hidden)
        out = self.fc(lstm_out[:, -1, :])
        return out, hidden

# 3. Prosty agent RNN (np. tylko LSTM na wektorach cech)
class SimpleRNN(nn.Module):
    def __init__(self, input_size, n_actions, hidden_size=128):
        super(SimpleRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_actions)
    def forward(self, x, hidden=None):
        # x: (batch, seq, features)
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        out = self.fc(lstm_out[:, -1, :])
        return out, hidden

# --- Przykładowe klasy agentów (DQN-style) ---

class CNNAgent:
    def __init__(self, input_shape, n_actions, device='cuda', config=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = CNN(input_shape, n_actions).to(self.device)
        self.target_model = CNN(input_shape, n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.buffer = ReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = 0.99
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
            state_v = torch.tensor(np.array([state]), device=self.device).float().permute(0,3,1,2)
            q_vals = self.model(state_v)
            _, action = torch.max(q_vals, dim=1)
            action = int(action.item())
        return ['left', 'right', 'up', 'down', 'space'][action]
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state, device=self.device).permute(0,3,1,2).float() / 255.0
        next_state = torch.tensor(next_state, device=self.device).permute(0,3,1,2).float() / 255.0
        action = torch.tensor([['left', 'right', 'up', 'down', 'space'].index(a) for a in action], device=self.device).long()
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

class CNNRNNAgent:
    def __init__(self, input_shape, n_actions, device='cuda', config=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = CNNRNN(input_shape, n_actions).to(self.device)
        self.target_model = CNNRNN(input_shape, n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.buffer = ReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = 0.99
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
            state_v = torch.tensor(np.array([state]), device=self.device).float().permute(0,3,1,2)
            q_vals, _ = self.model(state_v)
            _, action = torch.max(q_vals, dim=1)
            action = int(action.item())
        return ['left', 'right', 'up', 'down', 'space'][action]
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state, device=self.device).permute(0,3,1,2).float() / 255.0
        next_state = torch.tensor(next_state, device=self.device).permute(0,3,1,2).float() / 255.0
        action = torch.tensor([['left', 'right', 'up', 'down', 'space'].index(a) for a in action], device=self.device).long()
        reward = torch.tensor(reward, device=self.device).float()
        done = torch.tensor(done, device=self.device).float()
        q_values, _ = self.model(state)
        q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q, _ = self.target_model(next_state)
            next_q = next_q.max(1)[0]
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

class SimpleRNNAgent:
    def __init__(self, input_size, n_actions, device='cuda', config=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = SimpleRNN(input_size, n_actions).to(self.device)
        self.target_model = SimpleRNN(input_size, n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.buffer = ReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = 0.99
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
            state_v = torch.tensor(np.array([state]), device=self.device).float().permute(0,3,1,2)
            q_vals, _ = self.model(state_v)
            _, action = torch.max(q_vals, dim=1)
            action = int(action.item())
        return ['left', 'right', 'up', 'down', 'space'][action]
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state, device=self.device).permute(0,3,1,2).float() / 255.0
        next_state = torch.tensor(next_state, device=self.device).permute(0,3,1,2).float() / 255.0
        action = torch.tensor([['left', 'right', 'up', 'down', 'space'].index(a) for a in action], device=self.device).long()
        reward = torch.tensor(reward, device=self.device).float()
        done = torch.tensor(done, device=self.device).float()
        q_values, _ = self.model(state)
        q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q, _ = self.target_model(next_state)
            next_q = next_q.max(1)[0]
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

class DoubleDQNAgent:
    def __init__(self, input_shape, n_actions, device='cuda', config=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = CNN(input_shape, n_actions).to(self.device)
        self.target_model = CNN(input_shape, n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        # Wybór bufora: PER lub zwykły
        if config and config.get('per', False):
            self.buffer = PrioritizedReplayBuffer(config.get('buffer_size', 10000))
            self.use_per = True
        else:
            self.buffer = ReplayBuffer(config.get('buffer_size', 10000) if config else 10000)
            self.use_per = False
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.gamma = config.get('gamma', 0.99) if config else 0.99
        self.n_actions = n_actions
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learn_step = 0
        self.update_target_steps = 1000
        self.model_path = 'double_dqn_model.pth'
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict())
    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            state_v = torch.tensor(np.array([state]), device=self.device).float().permute(0,3,1,2)
            q_vals = self.model(state_v)
            _, action = torch.max(q_vals, dim=1)
            action = int(action.item())
        return ['left', 'right', 'up', 'down', 'space'][action]
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        if self.use_per:
            state, action, reward, next_state, done, indices, weights = self.buffer.sample(self.batch_size)
        else:
            state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
            weights = np.ones(self.batch_size, dtype=np.float32)
        state = torch.tensor(state, device=self.device).permute(0,3,1,2).float() / 255.0
        next_state = torch.tensor(next_state, device=self.device).permute(0,3,1,2).float() / 255.0
        # Popraw obsługę typu akcji (int lub string)
        if isinstance(action[0], (int, np.integer)):
            action_tensor = torch.tensor(action, device=self.device).long()
        else:
            action_tensor = torch.tensor([['left', 'right', 'up', 'down', 'space'].index(a) for a in action], device=self.device).long()
        reward = torch.tensor(reward, device=self.device).float()
        done = torch.tensor(done, device=self.device).float()
        weights = torch.tensor(weights, device=self.device).float()
        # Double DQN update
        with torch.no_grad():
            next_q_online = self.model(next_state)
            next_actions = next_q_online.argmax(1)
            next_q_target = self.target_model(next_state)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            expected_q = reward + self.gamma * next_q * (1 - done)
        q_values = self.model(state).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        loss = (weights * (q_values - expected_q) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # PER: aktualizacja priorytetów
        if self.use_per:
            td_errors = (q_values - expected_q).detach().cpu().numpy()
            self.buffer.update_priorities(indices, np.abs(td_errors) + 1e-6)
        self.learn_step += 1
        if self.learn_step % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
    def save(self):
        torch.save(self.model.state_dict(), self.model_path)

# Automatyczny wybór najlepszego agenta (na podstawie szybkości predykcji na dummy danych)
def get_best_agent(input_shape, n_actions, device='cuda', config=None):
    agent_type = config.get('type', 'DQN') if config else 'DQN'
    if agent_type == 'DoubleDQN':
        return DoubleDQNAgent(input_shape, n_actions, device=device, config=config)
    agents = [CNNAgent(input_shape, n_actions, device), CNNRNNAgent(input_shape, n_actions, device), SimpleRNNAgent(np.prod(input_shape), n_actions, device)]
    times = []
    for agent in agents:
        try:
            if isinstance(agent, CNNAgent):
                dummy = torch.zeros(1, *input_shape).to(agent.device)
                start = time.time()
                agent.model(dummy)
                times.append(time.time() - start)
            elif isinstance(agent, CNNRNNAgent):
                dummy = torch.zeros(1, 2, *input_shape).to(agent.device)  # seq=2
                start = time.time()
                agent.model(dummy)
                times.append(time.time() - start)
            elif isinstance(agent, SimpleRNNAgent):
                dummy = torch.zeros(1, 2, np.prod(input_shape)).to(agent.device)
                start = time.time()
                agent.model(dummy)
                times.append(time.time() - start)
        except Exception as e:
            times.append(float('inf'))
    best_idx = int(np.argmin(times))
    return agents[best_idx]

def pretrain_agent(agent, n_steps=100000, batch_size=64):
    sim = SimpleSpaceInvadersSim()
    steps = 0
    while steps < n_steps:
        batch = generate_simulated_batch(sim, batch_size)
        for state, action, reward, next_state, done in batch:
            agent.remember(state, action, reward, next_state, done)
        agent.update()
        steps += batch_size
        if steps % 10000 == 0:
            print(f'Pretraining: {steps}/{n_steps} kroków')
    print('Pretraining zakończony.') 