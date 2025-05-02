import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym.wrappers import FrameStack
from torchvision import transforms as T
from gym.spaces import Box
import numpy as np
from collections import deque

class FrameStack:
    def __init__(self, num_frames=4, frame_shape=(84, 84)):
        self.num_frames = num_frames
        self.frame_shape = frame_shape
        self.frames = deque(maxlen=num_frames)
        
    def add(self, frame):
        self.frames.append(frame)
        
    def get(self):
        while len(self.frames) < self.num_frames:
            self.frames.appendleft(self.frames[0])
        return np.stack(self.frames, axis=0)

class TransformObservation():
    def __init__(self, shape):
        self.shape = (shape, shape)
        self.observation_space = Box(
            low=0, high=255,
            shape=(1, *self.shape),
            dtype=np.uint8
        )
        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize(self.shape, antialias=True),
            T.Normalize(0, 255)
        ])
        
    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        observation = self.transform(observation).squeeze(0)
        return observation

class QNet(nn.Module):
    def __init__(self, n_actions):
        super(QNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc_val = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q

class ReplayBuffer:
    def __init__(self, capacity, n_step=3, gamma=0.99, alpha=0.5):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        
    def add(self, state, action, reward, next_state, done, episode_end=False):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        
        state_n, action_n, reward_n, next_state_n, done_n = self._get_n_step_info()
        
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state_n, action_n, reward_n, next_state_n, done_n))
        else:
            self.buffer[self.pos] = (state_n, action_n, reward_n, next_state_n, done_n)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        
        if episode_end or done == 1:
            while len(self.n_step_buffer) > 0:
                state_n, action_n, reward_n, next_state_n, done_n = self._get_n_step_info()
                
                if len(self.buffer) < self.capacity:
                    self.buffer.append((state_n, action_n, reward_n, next_state_n, done_n))
                else:
                    self.buffer[self.pos] = (state_n, action_n, reward_n, next_state_n, done_n)
                
                self.priorities[self.pos] = max_priority
                self.pos = (self.pos + 1) % self.capacity
                
                self.n_step_buffer.popleft()
            
            self.n_step_buffer.clear()
    
    def _get_n_step_info(self):
        reward, next_state, done = 0.0, None, None
        for idx, (_, _, r, next_s, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * r
            next_state = next_s
            done = d
            if d == 1:
                break
        state, action, _, _, _ = self.n_step_buffer[0]
        return state, action, reward, next_state, done
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            probs = self.priorities
        else:
            probs = self.priorities[:self.pos]
        
        probs = probs ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class DQNVariant:
    def __init__(self, action_size):
        self.action_size = action_size
        self.gamma = 0.99
        self.batch_size = 64
        self.learn_start = 120000
        
        self.action_range = 7
        self.max_x = np.ones(8)
        self.max_x_decay = 0.95
        
        self.q_update_freq = 4
        self.target_update_freq = 1000
        
        self.frame_count = 0
        self.update_count = 0
        
        self.epsilon = np.ones((8, 4000))
        self.eps_min = 0.01
        self.eps_decay = 0.001
        
        self.testing = False

        self.device = torch.device("cpu")

        self.q_net = QNet(action_size).to(self.device)
        self.target_net = QNet(action_size).to(self.device)
        self.update()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.00025)
        self.replay_buffer = ReplayBuffer(600000)

    def get_action(self, state, curr_xpos=None, last_xpos=None, stage=None):
        deterministic = True
        if(not self.testing):
            if(curr_xpos != last_xpos):
                self.epsilon[stage][last_xpos] = max(self.eps_min, self.epsilon[stage][last_xpos] - self.eps_decay)
            deterministic = random.random() > self.epsilon[stage][curr_xpos]

        if(not deterministic): 
            temp = 0.75 + max(1, (curr_xpos/self.max_x[stage]))
            state = torch.tensor(np.array(state).copy(), dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_net(state)
            limited_q_values = q_values[0, :self.action_range]
            scale_q = limited_q_values / temp
            action_probs = F.softmax(scale_q, dim=0)
            action = torch.multinomial(action_probs, 1).item() 
            if(action > 6):
                print("wtf")
            return action
        
        with torch.no_grad():
            state = torch.tensor(np.array(state).copy(), dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_net(state)
            action = torch.argmax(q_values).item()
            # print(action, q_values)
        return action

    def update(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)

    def train(self):
        if len(self.replay_buffer.buffer) < self.learn_start:
            return
        
        beta = min(0.4 + (self.update_count / 1e6), 1.0)
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size, beta)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(np.array(weights), dtype=torch.float32, device=self.device).unsqueeze(1)
        
        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = (q_values - target_q).squeeze(1)
        loss = F.smooth_l1_loss(q_values, target_q, reduction='none').squeeze(1)
        loss = (weights * loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, (td_errors.abs().cpu().detach().numpy() + 1e-6))
        
        self.update_count += 1
        if(self.update_count % self.target_update_freq == 0):
            self.update()

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        
        self.agent = DQNVariant(12)
        checkpoint = torch.load(f"dqn_agent.pth", map_location=self.agent.device)
        self.agent.q_net.load_state_dict(checkpoint['q_net'])
        self.agent.testing = True
        self.agent.q_net.eval()
        
        self.framestack = FrameStack()
        self.transform = TransformObservation(84)
        self.step = 0
        self.skip = 4
        self.action = 0

    def act(self, observation):
        self.transform.observation(observation)
        self.framestack.add(observation)
        if(self.step % self.skip == 0):
            state = torch.tensor(np.array(observation).copy(), dtype=torch.float32).unsqueeze(0).to(self.agent.device)
            q_values = self.agent.q_net(state)
            self.action = torch.argmax(q_values).item()
        self.step += 1
        return self.action