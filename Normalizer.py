import numpy as np

class Normalizer():


    def __init__(self, state_space):
        self.n = np.zeros(state_space)
        self.mean = np.zeros(state_space)
        self.mean_diff = np.zeros(state_space)
        self.var = np.zeros(state_space)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

