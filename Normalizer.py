import numpy as np

class Normalizer():

    def __init__(self,state_space):
        self.mean = np.zeros(state_space)
        self.std = np.zeros(state_space)
        self.n = np.zeros(state_space)
        self.mean_diff = np.zeros(state_space)


    def observe_renormalize(self,input_state):
        self.n +=1
        prev_mean = self.mean.copy()
        self.mean += (input_state - self.mean)/self.n
        self.mean_diff += (input_state - self.mean) * (input_state - prev_mean)
        self.std = np.sqrt((self.mean_diff / self.n).clip(min=1e-2))
        return (input_state-self.mean)/self.std


