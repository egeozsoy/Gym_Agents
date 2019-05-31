import gym
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from collections import namedtuple
from torch.distributions import Categorical
torch.manual_seed(7)

env = gym.make('MountainCar-v0')
env._max_episode_steps = 2000
env.reset()


# helpful reference https://github.com/ts1829/RL_Agent_Notebooks/blob/master/CartPole/Policy%20Gradient%20with%20Cartpole%20and%20PyTorch%20(Medium%20Version).ipynb

# Deep Q Learning Network that predicts rewards for every action
class DQN(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()
        self.linear = nn.Linear(state_space, 128)
        self.linear2 = nn.Linear(128, action_space)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x


Transition = namedtuple('Transition',
                        ('current_state', 'action', 'next_state', 'reward'))


# Function with the learning logic
def learn(game_memory, p_history):
    game_memory = Transition(*zip(*game_memory))
    rewards = []
    agg_reward = 0

    # calculate rewards from the back
    # This means if game was played for 20 steps, the 19.th step has reward 1, 18 reward 1 + 1*gamma ~ 1.99 etc.
    for reward in game_memory.reward[::-1]:
        agg_reward = reward + gamma * agg_reward
        rewards.insert(0, agg_reward)

    rewards = torch.FloatTensor(rewards)
    # normalize by diving to 4000
    rewards /= env._max_episode_steps

    # This is the most interesting part, instead of using a classic log function, we want pytorch to maximize the expected reward.
    # As pytorch minimizes the loss, we can define our problem as minimizing the negative of the expected reward(which is the same as maximizing the expected reward)
    # More about expected reward can be found here: https://medium.com/@jonathan_hui/rl-policy-gradients-explained-9b13b688b146
    loss = (torch.sum(torch.mul(p_history, Variable(rewards)).mul(-1), -1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


learning_rate = 0.01
# Discount Rate
gamma = 0.99
# Training Episodes
episodes = 5001
# Max Steps per episode
steps = env._max_episode_steps
state_space = 2
action_space = env.action_space.n

model = DQN(state_space, action_space)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
goal_reached_counter = 0
for episode in range(episodes):
    current_state = env.reset()

    game_memory = []
    policy_history = None
    for step in range(steps):
        if episode % 1000 == 0:
            env.render()

        current_state = Variable(torch.FloatTensor([current_state]))
        # let model calculate a policy for the current state
        Q = model(current_state)

        c = Categorical(Q)
        action = c.sample()

        next_state, reward, done, _ = env.step(action.item())

        action_prob = c.probs[0,action]
        if policy_history is None:
            policy_history = action_prob
        else:
            policy_history = torch.cat([policy_history, action_prob])

        # if we reach the goal before the steps, reward should be 1
        if done and step < steps - 1:
            goal_reached_counter += 1
            reward = 1000

        game_memory.append(Transition(current_state, action, next_state, reward))

        current_state = next_state

        if done:
            break

    if episode % 1000 == 0:
        env.close()

    # we learn after each game, instead of after each epoch
    learn(game_memory, policy_history)


    if episode % 50 == 0:
        print('Episode: {}, Goal reaching rate: {}'.format(episode,goal_reached_counter / 50))
        goal_reached_counter = 0
