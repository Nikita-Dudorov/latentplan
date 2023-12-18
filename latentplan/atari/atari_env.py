import atari_py
import cv2
import numpy as np
from collections import deque
import random

from .atari_dataset import create_atari_dataset


class AtariEnv():
    def __init__(self, args):
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game.lower()))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return state

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(np.zeros((84, 84), dtype=np.uint8))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return np.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = np.zeros((2, 84, 84), dtype=np.uint8)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done, info
        return np.stack(list(self.state_buffer), 0), reward, done, {}

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def get_dataset(self, debug=True, **kwargs):
        if debug:
            import pickle
            with open('/home/nikitad/projects/def-martin4/nikitad/decision-transformer/atari/dqn_replay/Breakout/atari_debug.pickle', 'rb') as f:
                dataset = pickle.load(f)
        else:
            num_buffers=50
            num_steps=500000
            game=self.game
            trajectories_per_buffer=10
            data_dir_prefix='/home/nikitad/projects/def-martin4/nikitad/decision-transformer/atari/dqn_replay/'
            dataset = create_atari_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer)
        return dataset

class AtariArgs:
    def __init__(self, game, seed):
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4

if __name__ == "__main__":
    args = AtariArgs(game='Breakout', seed=123)
    env = AtariEnv(args)
    s0 = env.reset()
    s, r, d, info = env.step(0)
    print(s.shape, s0.shape, r, d, info)
