import numpy as np
from utils import softmax
import Normalizer

class bb_numpy():

    def __init__(self,action_space,state_space,max_length,state_renormalize = False,continuous_action = False):
        self.action_space = action_space
        self.state_space = state_space
        self.max_length = max_length
        self.continuous_action = continuous_action
        self.state_renormalize = state_renormalize
        self.w_list = []
        self.b_list = []

    # def _init_model(self,action_space,state_space):


    def forward_propagate(self,x,init = True):
        if self.continuous_action == True:
            action = self._add_layer(x,self.state_space,self.action_space,0,activation = np.tanh,init = init)[0]
        else:
            action = self._add_layer(x,self.state_space,self.action_space,0,activation = softmax,init = init)
            action = np.argmax(action)
        return action

    def _add_layer(self,x,in_size,out_size,layer_id,activation = None,init = True):
        if init == True:
            w = np.random.randn(in_size,out_size)
            b = np.random.randn(1,out_size)
            self.w_list.append(w)
            self.b_list.append(b)
        else:
            w = self.w_list[layer_id]
            b = self.b_list[layer_id]

        x_w_b = x@w + b
        if activation == None:
            return x_w_b
        else:
            return activation(x_w_b)


    def get_param(self):
        return self.w_list,self.b_list

    def get_flat_param(self):
        param = []
        for i in range(len(self.w_list)):
            param += self.w_list[i].flatten().tolist()
            param += self.b_list[i].flatten().tolist()
        return param

    # def set_param(self):
        

    def set_flat_param(self,param):
        param = param.flatten()
        start_idx = 0
        for i in range(len(self.w_list)):

            self.w_list[i] = param[start_idx:start_idx + self.w_list[i].shape[0] * self.w_list[i].shape[1]]\
            .reshape((self.w_list[i].shape[0], self.w_list[i].shape[1]))
            start_idx += self.w_list[i].shape[0] * self.w_list[i].shape[1]
            self.b_list[i] = param[start_idx:start_idx + self.b_list[i].shape[0] * self.b_list[i].shape[1]]\
            .reshape((self.b_list[i].shape[0], self.b_list[i].shape[1]))
            start_idx += self.b_list[i].shape[0] * self.b_list[i].shape[1] 

    def roll_out(self,param,env,normalizer,render = False, init = False):
        self.set_flat_param(param)

        state = env.reset()
        done = False
        state = env.reset()
        state = state[np.newaxis,...]
        individual_fit = 0

        counter = 0
        while not done:
            #render the env
            if render == True:
                env.render()
            else:
                pass

            action = self.forward_propagate(state,init = init)
            state,reward,done, _ = env.step(action)
            if self.state_renormalize == True:
                state = normalizer.observe_renormalize(state)
            state = state[np.newaxis,...]
            individual_fit += reward

            counter +=0
            if counter > self.max_length:
                break

        env.close()
        return individual_fit

        





