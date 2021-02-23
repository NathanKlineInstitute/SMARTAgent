'''
TODO

-visually check the 8 directions inputs
-ball is lost on top of the screen and when hitting th rackets.
-make sure propagation is fluid
-voting system for the 3 motors
-reward: PostPre nu change or MSTDP?
-connect reward process

'''

import torch, cv2, bindsnet, time
import matplotlib.pyplot as plt
import matplotlib
import argparse
from bindsnet.analysis.plotting import plot_spikes, plot_input
from bindsnet.environment import GymEnvironment
from nki_network_v2 import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--v1_neurons', type=int, default=400)
parser.add_argument('--m_neurons', type=int, default=32)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--num_episodes', type=int, default=100)
parser.add_argument('--runtime', type=int, default=100)
parser.add_argument('--plot_interval', type=int, default=10)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=True, render=True, gpu=True)

args = parser.parse_args()

seed = args.seed
v1_neurons = args.v1_neurons
m_neurons = args.m_neurons
dt = args.dt
num_episodes = args.num_episodes
runtime = args.runtime
plot_interval = args.plot_interval
gpu = args.gpu
render = args.render

directions_intensity = 200.
frame_intensity = 200.

device = 'cpu'
if gpu and torch.cuda.is_available():
    #  torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
    device = 'cuda'
    print('CUDA used.')
else:
    torch.manual_seed(seed)

print('Building network...')
network, spikes = Return_nki_net(
    v1_neurons=v1_neurons,
    m_neurons=m_neurons,
    dt=dt, device=device,
    runtime=runtime
)
network.to(device)
print(bindsnet.analysis.visualization.summary(network))

# -----------------------------------------------------------------------
from bindsnet.encoding import poisson
import itertools
import numpy as np
import sys

colinearity = torch.zeros((8, 2), dtype=torch.float32)
directionals = torch.zeros((8, 80, 80), dtype=torch.float32)

unseen_ball = 0
old_ball_x, old_ball_y = 0, 0

# get ball position from a 80x80 screen
def get_ball_pos(img):
    global unseen_ball, old_ball_x, old_ball_y
    i = img[:, 10:70].numpy()  # filter out racket zones
    if i.sum() == 0:  # ball not visible ? it will be in the center, soon?
        if unseen_ball <= 0:
            x, y = 40, 40
        else:
            unseen_ball -= 1
            x = old_ball_x
            y = old_ball_y
    else:
        unseen_ball = 3
        pos = i.argmax()
        y = pos // 60
        x = pos % 60
        old_ball_x = x
        old_ball_y = y
    return x+10, y


# get rackets vertical positions
def get_rackets(img):
    im_atari = img[:, 0:10].numpy()
    im_player = img[:, 70:80].numpy()
    atari_y = np.argmax(im_atari) // 10
    player_y = np.argmax(im_player) // 10
    return atari_y, player_y


# simplifies atari screen
def simplify_frame(img):
    img = img[34:194]  # crop
    img = img[::2, ::2, 0]  # every 2 pixel, monochrome from red
    img[img == 144] = 0  # erase background (background type 1)
    img[img == 109] = 0  # erase background (background type 2)
    img[img != 0] = 1.  # all colors=1
    return img


# down_sample linearly 4x
def down_sample4x(img):
    img2 = torch.zeros((img.shape[0] // 4, img.shape[0] // 4))
    for i in range(4):
        for j in range(4):
            img2 += img[i::4, j::4]
    img2 *= 0.0625
    return img2


# returns positive of directional diff
def dir_diff(dx, dy, a, b):
    c = torch.zeros(80, 80)
    c[1:79, 1:79] = a[1:79, 1:79] - b[1+dx:79+dx, 1+dy:79+dy]
    return torch.nn.functional.relu(down_sample4x(c))


# action policy
# !!!! action_pop_size and total_actions are not defineds

def policy(rspikes, eps):
    q_values = torch.Tensor([rspikes[(i * action_pop_size):(i * action_pop_size) + action_pop_size].sum()
                             for i in range(total_actions)])
    A = np.ones(4, dtype=float) * eps / 4
    if torch.max(q_values) == 0:
        return [0, 1, 0, 0]
    best_action = torch.argmax(q_values)
    A[best_action] += (1.0 - eps)
    return A


for i in range(8):
    a = i*2*np.pi/8
    colinearity[i, 0] = np.sin(a)
    colinearity[i, 1] = np.cos(a)


# convert vector coordinates into the 8 direction maps
def dir_vector(x, y, dx, dy):
    for m in range(8):
        result = max(colinearity[m, 0]*dx + colinearity[m, 1]*dy, 0.0)
        directionals[m, x, y] = result
    return 0


total_t = 0
episode_rewards = np.zeros(num_episodes)
q_spikes = []
epsilon = 0.0  # probability of picking random action
spike_ims, spike_axes = None, None
inpt_axes, inpt_ims = None, None
matplotlib.use('TkAgg')

# Load Breakout environment.
for i_episode in range(num_episodes):
    # Load SpaceInvaders environment.
    environment = GymEnvironment('Pong-v0')

    environment.reset()
    for warmup in range(20):
        environment.step(0)

    obs, _, _, _ = environment.step(0)

    done = False
    duration = 0
    reward = total_reward = 0
    action = 0
    refrac = 0
    ball_x, ball_y = 40, 40
    atari_y, player_y = 40, 40
    start_time = time.time()
    while not done:
        directionals *= 0.

        old_ball_x = ball_x
        old_ball_y = ball_y

        old_atari_y = atari_y
        old_player_y = player_y

        prev_reward = reward
        obs, reward, done, info = environment.step(action)
        environment.render()
        #time.sleep(0.020)

        obs = simplify_frame(obs[0])
        ball_x, ball_y = get_ball_pos(obs)
        obs[ball_y, ball_x] = 5.0  # a darker ball !
        atari_y, player_y = get_rackets(obs)

        ball_speed_x = ball_x - old_ball_x
        ball_speed_y = ball_y - old_ball_y
        dir_vector(ball_x, ball_y, ball_speed_x, ball_speed_y)

        atari_speed_y = atari_y - old_atari_y
        dir_vector(5, atari_y, 0, atari_speed_y)

        player_speed_y = player_y - old_player_y
        dir_vector(75, player_y, 0, player_speed_y)

        dirs = torch.zeros(8*20, 20, dtype=torch.float32)
        for i in range(8):
            dirs[i*20:(i+1)*20, :] += down_sample4x(directionals[i])

        dirs = torch.flatten(dirs)
        dirs += 1e-12
        dirs = dirs.unsqueeze(dim=0)

        obs = torch.flatten(down_sample4x(obs))
        obs = obs.unsqueeze(dim=0)

        network.reset_state_variables()
        network.run(
            inputs={'Thalamus_Input': poisson(frame_intensity*obs, time=runtime, dt=dt).to(device),
                    'V1_Directions_Input': poisson(directions_intensity*dirs, time=runtime, dt=dt).to(device)},
            time=runtime,
            reward=1.0 if prev_reward < reward else 0.0
        )
        tata = spikes['V1_Exc'].get("s")
        #momo = torch.sum(tata)
        #print(momo)

        action = 0
        refrac -= 1
        if refrac <= 0:
            delta = ball_y - 3 - player_y
            if delta > 2:
                action = 3
                refrac = np.random.randint(0, 4)
            if delta < -2:
                action = 4
                refrac = np.random.randint(0, 4)
        duration += 1
        avg_time = (time.time() - start_time) / duration
        print('time=', duration, 'fps=', 1. / avg_time)


        if duration%5 == 1 or True:
            #spikes_ = {'V1_Exc': spikes['V1_Exc'].get("s")}
            #spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)

            #inpt_axes, inpt_ims = plot_input(
            #    image=frame_intensity*obs.view(20, 20), inpt=poisson(frame_intensity*obs, time=runtime, dt=dt).sum(0).view(20, 20), axes=inpt_axes, ims=inpt_ims
            #"")
            tyty = tata.sum(axis=0).view(20, 20)
            inpt_axes, inpt_ims = plot_input(
                image=frame_intensity*obs.view(20, 20), inpt=tyty, axes=inpt_axes, ims=inpt_ims
            )
            print(tyty.sum())

            plt.show()
            plt.pause(.10)
        total_reward += reward

    print(duration, total_reward)
    '''

    for t in itertools.count():
        # print(t)
        network.reset_state_variables()
        old_action = action
        obs = obs.unsqueeze(dim=0)  # add batch dim
        network.run(
            inputs={'X': poisson(obs, time=runtime, dt=dt)},
            time=runtime,
            reward=0.
        )
        readout_spikes = network.monitors['M'].get('s')

        # print(torch.sum(readout_spikes, dim=0))
        action_probs = policy(torch.sum(readout_spikes, dim=1), epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        next_obs, reward, done, info = environment.step(VALID_ACTIONS[action])
        next_obs = simplify_frame(next_obs[0])

        reward = np.sign(reward)
        episode_rewards[i_episode] += reward
        q_spikes.append(torch.sum(readout_spikes, dim=0))
        total_t += 1

        if t > 1000 or done:
            print('\rStep {} ({}) @ Episode {}/{}'.format(t, total_t, i_episode + 1, num_episodes), end='')
            print('\nEpisode Reward: {}'.format(episode_rewards[i_episode]))
            sys.stdout.flush()
            break

        obs = next_obs
    '''