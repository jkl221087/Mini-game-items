import pygame
import numpy as np
import gym
from gym import spaces
import random
import math
import csv
import matplotlib.pyplot as plt
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

class Aitank(gym.Env):
    def __init__(self):
        super(Aitank, self).__init__()
        
        pygame.init()

        # 屏幕设置
        self.screen_width = 800
        self.screen_height = 600
        self.win = pygame.display.set_mode((self.screen_width, self.screen_height))

        # 坦克设置
        self.tank_width = 50
        self.tank_height = 30
        self.moving_tank_x = self.screen_width // 2 # 400
        self.moving_tank_y = self.screen_height - 50

        # 导弹设置
        self.missiles = []
        self.missile_speed = 2
        self.missile_radius = 5

        self.action_space = spaces.Discrete(3)


        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1 + 2 * 10,), dtype=np.float32)

        self.running = True
        self.clock = pygame.time.Clock()

    def reset(self):
        self.moving_tank_x = self.screen_width // 2
        self.missiles = []
        return self._get_state()

    def step(self, action):
        self.tank1_moving(action)
        self._update_missiles()
        
        reward = 1  # 每一步都给予小的正奖励
        done = False

        # 检查碰撞
        for missile in self.missiles:
            if self._check_collision(missile):
                reward = -100 
                done = True
                break

        return self._get_state(), reward, done, {}

    def _get_state(self):
    # 坦克位置
        tank_relative_x = self.moving_tank_x / self.screen_width

        # 只考慮最近的7枚導彈
        closest_missiles = sorted(self.missiles, key=self._calculate_distance)[:7]

        state = [tank_relative_x]

        for missile in closest_missiles:
            # 導彈的相對位置
            missile_relative_x = missile[0] / self.screen_width
            missile_relative_y = missile[1] / self.screen_height

            # 計算導彈與坦克的相對距離和方向
            dx = missile[0] - (self.moving_tank_x + self.tank_width / 2)
            dy = missile[1] - (self.moving_tank_y + self.tank_height / 2)
            distance = math.sqrt(dx**2 + dy**2) / math.sqrt(self.screen_width**2 + self.screen_height**2)
            angle = math.atan2(dy, dx) / math.pi  # 範圍：-1 到 1

            state.extend([missile_relative_x, missile_relative_y, distance, angle])

        # 如果導彈少於7枚，用0填充
        state = state[:21]
        while len(state) < 21:
            state.append(0.0)

        return np.array(state)

    def _check_collision(self, missile):
        return (self.moving_tank_x < missile[0] < self.moving_tank_x + self.tank_width and
                self.moving_tank_y < missile[1] < self.moving_tank_y + self.tank_height)

    def _update_missiles(self):
       
        self.missiles = [[x, y + self.missile_speed] for x, y in self.missiles if y < self.screen_height]

        # 随机生成新的导弹
        if random.random() < 0.07: 
            self.missiles.append([random.randint(0, self.screen_width), 0])

    def tank1_moving(self, action):
        if action == 1:
            self.moving_tank_x = max(0, self.moving_tank_x - 5)
        elif action == 2:
            self.moving_tank_x = min(self.screen_width - self.tank_width, self.moving_tank_x + 5)

    def _calculate_distance(self, missile):
        tank_center_x = self.moving_tank_x + self.tank_width // 2
        tank_center_y = self.moving_tank_y + self.tank_height // 2

        distance_x = tank_center_x - missile[0]
        distance_y = tank_center_y - missile[1]
        return math.hypot(distance_x, distance_y)

    def _get_nearest_missile(self):
        if self.missiles:
            return min(self.missiles, key=self._calculate_distance)
        return None

    def render(self):
        self.win.fill((255, 255, 255))

        
        pygame.draw.rect(self.win, (0, 255, 0), (self.moving_tank_x, self.moving_tank_y, self.tank_width, self.tank_height))

        nearest_missile = self._get_nearest_missile()

        for missile in self.missiles:
            color = (255, 0, 0)
            line_color = (0, 0, 255) 
            if missile == nearest_missile:
                color = (255, 165, 0)  
                line_color = (255, 165, 0) 

            pygame.draw.circle(self.win, color, (int(missile[0]), int(missile[1])), self.missile_radius)
            
            
            start_pos = (self.moving_tank_x + self.tank_width // 2, self.moving_tank_y)
            end_pos = (int(missile[0]), int(missile[1]))
            pygame.draw.line(self.win, line_color, start_pos, end_pos, 1)

        pygame.display.flip()

    def close(self):
        pygame.quit()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = []
        self.batch_size = 32

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # print(f"State shape: {state.shape}") 
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)


        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.q_network.state_dict())

def train():
    env = Aitank()
    agent = DQNAgent(state_dim=21, action_dim=3)  # 21是狀態維度，3是動作維度
    total_episodes = 1000
    rewards = []
    start_episode = 0

    # 檢查是否有先前的訓練數據
    if os.path.exists('training_state.pkl'):
        with open('training_state.pkl', 'rb') as f:
            state_dict = pickle.load(f)
            start_episode = state_dict['episode']
            rewards = state_dict['rewards']
            agent.load('dqn_model.pth')  
            print(f"Resuming training from episode {start_episode + 1}")
    
    for episode in range(start_episode, total_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            total_reward += reward
            state = next_state
            
            env.render()
            env.clock.tick(30)
        
        rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        # 每 100 個 episode 保存一次狀態
        if (episode + 1) % 100 == 0:
            save_state(episode + 1, rewards, env, agent)
            agent.update_target_network()
    
    env.close()
    
    # 最後一次保存狀態
    save_state(total_episodes, rewards, env, agent)
    
    # 繪製並保存獎勵曲線
    plot_rewards(rewards)

def save_state(episode, rewards, env, agent):
    state_dict = {
        'episode': episode,
        'rewards': rewards,
    }
    with open('training_state.pkl', 'wb') as f:
        pickle.dump(state_dict, f)
    
    agent.save('dqn_model.pth')  # 保存模型
    
    # 保存訓練結果到 CSV 文件
    with open('training_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total Reward"])
        for i, reward in enumerate(rewards):
            writer.writerow([i+1, reward])

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards)+1), rewards)
    plt.title('Training Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('reward_curve.png')
    plt.close()

if __name__ == "__main__":
    train()
