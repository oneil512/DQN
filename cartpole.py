import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


BATCH_SIZE = 32
GAMMA = 0.99

class Memory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_actions, n_state):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(n_state, 64)
        self.l2 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        return x


class Agent():
    def __init__(self, n_actions, n_states):
        self.steps_done = 0
        self.n_actions = n_actions
        self.n_states = n_states
        self.memory = Memory(capacity=10000)
        self.policy_net = DQN(n_actions, n_states).to(device)
        self.target_net = DQN(n_actions, n_states).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=.01)

        self.target_net.eval()

    def select_action(self, state):
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                res = self.policy_net(state)

                return res.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

rewards = []
class Environment():
    def __init__(self, problem):
        self.env = gym.make(problem).unwrapped

    def run(self, agent):
        R = 0
        self.env.reset()
        s = torch.tensor(self.env.state, dtype=torch.float32).view(-1, agent.n_states)
        count = 0
        while True:
            count += 1
            a = agent.select_action(s)
            s_, r, done, _ = self.env.step(a.item())
            s_ = torch.tensor(s_, dtype=torch.float32).view(-1, agent.n_states)
            if done:
                s_ = None
            R += r
            r = torch.tensor(r, dtype=torch.float32).view(1)
            agent.memory.push(s, a, s_, r)
            s = s_
            agent.replay()
            
            if done:
                rewards.append(R)
                plot(rewards)
                print('reward', R)
                break


problem = 'CartPole-v0'
env = Environment(problem)
n_actions = env.env.action_space.n
n_states = env.env.observation_space.shape[0]
agent = Agent(n_actions, n_states)
TARGET_UPDATE = 5

def plot(rewards):
    plt.figure(1)
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('rewards')

    plt.plot(rewards)
    plt.pause(.0001)

episodes = 1000
for e in range(episodes):
    env.run(agent)

    if e % TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

plt.show()
print('Done')