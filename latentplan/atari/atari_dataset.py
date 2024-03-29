import numpy as np
import torch

from .fixed_replay_buffer import FixedReplayBuffer
from .vae import VAE


def create_atari_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
    # -- load data from memory (make more efficient)
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0
    while len(obss) < num_steps:
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]
        print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + game + '/1/replay_logs',
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000)
        if frb._loaded_buffers:
            done = False
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer
            while not done:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
                obss += [states]
                actions += [ac[0]]
                stepwise_returns += [ret[0]]
                if terminal[0]:
                    done_idxs += [len(obss)]
                    returns += [0]
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                returns[-1] += ret[0]
                i += 1
                if i >= 100000:
                    obss = obss[:curr_num_transitions]
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0
                    i = transitions_per_buffer[buffer_num]
                    done = True
            num_trajectories += (trajectories_per_buffer - trajectories_to_load)
            transitions_per_buffer[buffer_num] = i
        print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    # pre-process for TAP
    assert len(obss) == len(actions) == len(stepwise_returns) == len(rtg)
    terminals = np.zeros_like(stepwise_returns, dtype=bool)
    terminals[done_idxs-1] = 1 
    timeouts = np.zeros_like(terminals, dtype=bool)
    dataset = {
        'observations': obss, 
        'actions': actions, 
        'rewards': stepwise_returns, 
        'terminals': terminals,
        'timeouts': timeouts,
    }
    # import pickle
    # with open('atari_debug.pickle', 'wb') as f:
    #     pickle.dump(dataset, f)
    return dataset


def atari_obs_embed(observations, device):
    checkpoint_path = '/home/nikitad/projects/def-martin4/nikitad/vae_checkpoints/VAEmodel_40.pkl'
    latent_dim = 512
    channels = 4
    im_size = 84
    b_size = 128
    obs_encoder = VAE(zsize=latent_dim, channels=channels, imsize=im_size)
    obs_encoder.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
    obs_encoder.to(device)
    obs_encoder.eval()

    if observations.ndim != 4:  # add batch axis
        observations = observations.reshape((1,) + observations.shape) 
    n_oob = len(observations) % b_size
    b = torch.from_numpy(observations[-n_oob:]).to(device)
    obs_emb = obs_encoder.get_latent(b).cpu().numpy()
    for i in range(len(observations) // b_size):
        b = torch.from_numpy(observations[n_oob + i*b_size : n_oob + (i+1)*b_size]).to(device)
        embeddings = obs_encoder.get_latent(b).cpu().numpy()
        obs_emb = np.vstack((obs_emb, embeddings))

    return obs_emb.squeeze()