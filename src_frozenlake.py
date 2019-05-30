import gym
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from collections import namedtuple
import random

env = gym.make('FrozenLake-v0')
# env = gym.make('FrozenLake8x8-v0')
env.reset()

# helpful reference https://github.com/ts1829/RL_Agent_Notebooks/blob/master/frozenlake-v0%20PyTorch.ipynb

def OH(x, l):
    return torch.eye(l)[x,:]

class DQN(nn.Module):
    def __init__(self,state_space,action_space):
        super(DQN, self).__init__()
        self.linear = nn.Linear(state_space,action_space)

    def forward(self, x):
        return self.linear(x)

Transition = namedtuple('Transition',
                        ('current_state', 'action', 'next_state', 'reward'))

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def learn(replay_memory):
    if len(replay_memory) < batch_size:
        return
    transitions = replay_memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    current_state_batch = torch.cat(batch.current_state)
    reward_batch = torch.FloatTensor(batch.reward)

    # let model calculate a policy for the current state
    Q = model(Variable(current_state_batch))
    # let model calculate a policy for the next state
    Q_next = model(Variable(OH(batch.next_state, state_space)))
    # calculate the index of the maximum value
    maxQ_next, _ = torch.max(Q_next.data, 1)

    # lets calculate the our target
    targetQ = Variable(Q.data.clone(), requires_grad=False)
    # Adjust target because now we now what that move actually brings, use the current reward and the future predicted reward for this
    targetQ[0, batch.action] = reward_batch + torch.mul(maxQ_next, gamma)

    loss = loss_fnc(Q, targetQ)

    model.zero_grad()
    loss.backward()
    optimizer.step()


# Chance of random action
e = 0.1
learning_rate = 0.001
# Discount Rate
gamma = 0.99
# Training Episodes
episodes = 100001
# Max Steps per episode
steps = 99
batch_size = 10
state_space = env.observation_space.n
action_space = env.action_space.n

model = DQN(state_space,action_space)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

loss_fnc = nn.MSELoss()
rewards = 0
replay_memory = ReplayMemory(1000)
for episode in range(episodes):
    current_state = int(env.reset())

    for step in range(steps):

        current_state = Variable(OH([current_state], state_space))
        # let model calculate a policy for the current state
        Q = model(current_state)

        if np.random.rand(1) < e:
            action = env.action_space.sample()

        else:
            # gets the best action
            _,action = torch.max(Q,1)
            action = action.item()

        next_state, reward, done, _ = env.step(action)

        if episode % 50000 == 0:
            env.render()

        if done and reward == 0.0:
            reward = -1

        replay_memory.push(current_state,action,next_state,reward)

        current_state = next_state
        rewards += reward

        # train here
        if episode % batch_size == 0 and episode != 0:
            learn(replay_memory)

        if done:
            if reward > 0:
                e = 1. / ((episode / 50) + 10)
            break

    if episode % 500 == 0:
        print(rewards)
        rewards = 0