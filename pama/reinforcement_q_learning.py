from __future__ import print_function

import time

import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image

import math
import torch
import torch.optim as optim
import torchvision.transforms as T

from module import *
from mygame import *
from graphic import *

# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display
#
# plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
game = Pacman()

######################################################################
# Training
# --------

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
print('init net done')

steps_done = 0

episode_durations = []

#
# def plot_durations():
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())
#
#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################


def select_action(game):
    if game.gameover():
        return
    id = game.select_action()
    state,legal = game.get_state()
    state_ = torch.tensor([state], device=device)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            outout = np.asarray(policy_net(state_)[0])
            out = (legal*outout)
            for i in range(4):
                if legal[i] == 0:
                    out[i] = -1000
            id = out.argmax()
            return id #policy_net(state_).max(1)[1].view(1, 1).item()
    else:
        return id

######################################################################



num_episodes = 50
print("START TRAINING")

# g = Graphic(block_size=30)
num_episodes = 1000
for i_episode in range(num_episodes):
    # Initialize the environment and state

    game.reset(nb_bigdot=2)
    state, _ = game.get_state()

    # g.render(state)
    print("START epsisode %d",i_episode)
    for t in count():
        # Select and perform an action
        action = select_action(game)
        next_state, reward, done, _ = game.make_action(action)
        # g.render(next_state)
        reward_ = torch.tensor([reward], device=device).float()
        state_ = torch.tensor([state], device=device)
        action_ = torch.tensor([[np.long(np.argmax(action))]], device=device)
        next_state_ = torch.tensor([next_state], device=device)

        if done :
            #time.sleep(3)
            next_state = None
        # Store the transition in memory
        memory.push(state_, action_, next_state_, reward_)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        #print(t)
        optimize_model()
        if done:
            log = open("log4.txt", "a")

            print('Episode %d : reward %d < mydot %d:%d'%(i_episode ,reward, game.my_dot ,game.get_total_dot()) , file = log)
            print('Episode %d : reward %d < mydot %d:%d'%(i_episode ,reward, game.my_dot ,game.get_total_dot()))
            episode_durations.append(t + 1)
            # if (i_episode%100 == 0):
            #     plot_durations()
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

# print('Complete')
# plt.ioff()
# plt.show()