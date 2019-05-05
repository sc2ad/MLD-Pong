import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

from two_player_pong_env import PongEnv
from collections import namedtuple
from itertools import count

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

FRAME_COUNT = 2500

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class ReplayMemory(object):

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

class DQN(nn.Module):

    def __init__(self, state_size, outputs):
        super(DQN, self).__init__()
        """self.conv1 = nn.Conv2d(7, 14, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(14)
        self.conv2 = nn.Conv2d(14, 28, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(28)
        self.conv3 = nn.Conv2d(28, 28, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(28)"""
        linear_input_size = state_size
        self.fc1 = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))"""
        return self.fc1(x)

# TODO fill out the methods of this class
class PongPlayer(object):

    def __init__(self, save_path, load=False):
        self.save_path = save_path

        self.BATCH_SIZE = 2
        self.GAMMA = 0.999
        self.EPS_START = 0.95
        self.EPS_END = 0.05
        self.EPS_DECAY = 100
        self.TARGET_UPDATE = 10

        # Get number of actions from gym action space
        self.n_actions = 3

        self.policy_net = DQN(7, self.n_actions)
        self.target_net = DQN(7, self.n_actions)
        self.policy_net.double()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(10000)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.steps_done = 0

        if load:
            self.load()

    def get_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return torch.tensor([[torch.argmax(self.policy_net(torch.from_numpy(np.array(state))))]])
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([torch.tensor(s) for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch.reshape((self.BATCH_SIZE, 7))).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states.float().reshape((self.BATCH_SIZE, 7))).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, player2):
        num_episodes = 50
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            env = PongEnv()
            state1, state2 = env.reset()
            done = False
            print("Episode " + str(i_episode) + " has started")
            frames = 0
            while not done and frames < FRAME_COUNT:
                # Select and perform an action
                action1 = self.get_action(state1)
                action2 = player2.get_action(state2)
                next_state1, next_state2, reward, _, done, _ = env.step(action1.item(), action2)
                reward = torch.tensor([reward], device=device)

                # Store the transition in memory
                self.memory.push(torch.tensor(state1).double(), 
                    torch.tensor(action1), torch.tensor(next_state1).double(), torch.tensor(reward))

                # Move to the next state
                state1 = next_state1
                state2 = next_state2

                # Perform one step of the optimization (on the target network)
                # print("Beginning Optimization")
                self.optimize_model()
                # print("Completing Optimization")
                frames += 1

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            print("Episode " + str(i_episode) + " is finished")

        print('Complete')

    def reset(self):
        # TODO: this method will be called whenever a game finishes
        # so if you create a model that has state you should reset it here
        # NOTE: this is optional and only if you need it for your model
        pass

    def load(self):
        state = torch.load(self.save_path)
        self.policy_net.load_state_dict(state['policy_net_dict'])
        self.target_net.load_state_dict(state['target_net_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.memory = state['memory']
        self.BATCH_SIZE = state['BATCH_SIZE']

    def save(self):
        state = {
            'policy_net_dict': self.policy_net.state_dict(),
            'target_net_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory' : self.memory,
            'BATCH_SIZE' : self.BATCH_SIZE,
        }
        torch.save(state, self.save_path)

class PongBot(object):
    def get_action(self, state):
        if (state[1] > state[4]):
            return 2
        elif (state[1] < state[4]):
            return 0
        else:
            return 1

def play_game(player1, player2, render=True):
    # call this function to run your model on the environment
    # and see how it does
    env = PongEnv()
    state1, state2 = env.reset()
    action1 = player1.get_action(state1)
    action2 = player2.get_action(state2)
    done = False
    total_reward1, total_reward2 = 0, 0
    frames = 0
    while not done and frames < FRAME_COUNT:
        next_state1, next_state2, reward1, reward2, done, _ = env.step(action1.item(), action2)
        if render:
            env.render()
        action1 = player1.get_action(next_state1)
        action2 = player2.get_action(next_state2)
        total_reward1 += reward1
        total_reward2 += reward2
        frames += 1

    env.close()

Version1 = PongPlayer("./data.dat", load=True)
# Version1.train(PongBot())
# Version1.save()

for i in range(10):
    play_game(Version1, PongBot())
