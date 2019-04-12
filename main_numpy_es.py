from bb_numpy_net import bb_numpy
import multiprocessing as mp
from multiprocessing import Queue, Process
import os
import matplotlib.pyplot as plt
import gym
import numpy as np
from utils import SGD, get_info_summary,get_guassian_matrices
from Normalizer import Normalizer
import json
import time
import sys

class train_ES():
    def __init__(
                self,
                iterations = 20000,
                num_perturbations = 64,
                # env = 'BipedalWalker-v2',
                # env = 'CartPole-v1',
                # env = 'MountainCarContinuous-v0',
                env = 'MountainCar-v0',
                gamma = 0.99,
                sigma = 1,
                lr = 3 * 1e-2,
                max_length = 2000,
                num_test = 1,
                continuous_action = False,
                best = 32,
                reward_renormalize = False,
                state_renormalize = False,
                num_cpu = 5
                ):

        self.iterations = iterations
        self.num_perturbations = num_perturbations
        self.env = env
        self.gamma = gamma
        self.sigma = sigma
        self.lr = lr
        self.max_length = max_length
        self.num_test = num_test
        self.continuous_action = continuous_action
        self.best = best
        self.reward_renormalize = reward_renormalize
        self.state_renormalize = state_renormalize
        self.num_cpu = num_cpu


        global param_dir 
        param_dir = 'Param'
        global path 
        path = param_dir + '/' + env
        global name
        name = 'data.json'
        global param_dict
        param_dict = {}


    def _do_work(self,queue,bbnp,param,action_space,state_space,max_length,seed_id,num_workers,env,continuous_action):
        start = time.time()
        np.random.seed(seed_id)

        noises = self.sigma * np.random.randn(num_workers,len(param))
        noisy_param = param + noises
        
        fittness = []
        anti_fittness = []
        worker_summary = {}


        for ind in noisy_param:

            #do the roll out
            if self.state_renormalize == True:
                normal = Normalizer(state_space)
                ind_fit = bbnp.roll_out(ind,env,normal,render = False,state_renormalize = True)
                normal = Normalizer(state_space)
                ind_fit_anti = bbnw.roll_out(-ind,env,normal,render = False,state_renormalize = True)
            else:
                normal = Normalizer(state_space)
                ind_fit = bbnp.roll_out(ind,env,normal,render = False,init = False)
                ind_fit_anti = bbnp.roll_out(-ind,env,normal,render = False,init = False)


            fittness.append(ind_fit)
            anti_fittness.append(ind_fit_anti)

        end = time.time()
        # print('process id:{}  |  queue time:{}  |  seed_id:{}'.format(os.getpid(),end-start,seed_id))

        worker_summary['fit'] = fittness
        worker_summary['anti_fit'] = anti_fittness
        worker_summary['seed_id'] = seed_id
        queue.put(worker_summary)

    def train(self):

        #need to put into function
        
        if not os.path.exists("./" + param_dir):
            os.mkdir(param_dir)

        for test in range(self.num_test):
            fit_list = []
            iteration_list = []

            plt.ion()
            plt.show()

            #set up environment
            env = gym.make(self.env)

            if self.continuous_action:
                action_space = env.action_space.shape[0]
            else:
                action_space = env.action_space.n
            state_space = env.observation_space.low.size

            param_dict = {}
            #get param for the len
            best_reward = 0


            bbnp = bb_numpy(action_space,state_space,self.max_length,continuous_action = self.continuous_action,state_renormalize = self.state_renormalize)
            state = np.zeros(state_space)[None,...]
            bbnp.forward_propagate(state,init = True)

            #initialize param
            param = np.array(bbnp.get_flat_param())
            SGD_ = SGD(param, self.lr)

            for iteration in range(self.iterations):
                if self.num_perturbations % self.num_cpu != 0:
                    seed_id =  np.random.randint(np.iinfo(np.int32(10)).max, size=self.num_cpu + 1)
                else:
                    seed_id = np.random.randint(np.iinfo(np.int32(10)).max, size=self.num_cpu)

                queue = Queue()

                num_workers = [int(self.num_perturbations / self.num_cpu)] * self.num_cpu + [self.num_perturbations % self.num_cpu]
                workers = [Process(target = self._do_work,args = (queue,bbnp,param,action_space,state_space,self.max_length,seed_id[i],num_workers[i],env,self.continuous_action))\
                for i in range(len(seed_id))]

                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join()

                results = [queue.get() for p in workers]
                pert_fittness,anti_pert_fittness,seed_id = get_info_summary(results)
                print('pert_fit',pert_fittness,'\n','anti_pert_fit',anti_pert_fittness,'\n','seed_id',seed_id)
                pert_fittness = np.array(pert_fittness)[...,None]
                anti_pert_fittness = np.array(anti_pert_fittness)[...,None]
                noises = get_guassian_matrices(seed_id,num_workers,len(param),self.sigma)

                #record average_fit
                average_fit = np.sum(pert_fittness)/self.num_perturbations

                fit_list.append(average_fit)
                iteration_list.append(iteration)

                #dynamic plot the graph
                plt.plot(iteration_list,fit_list,'r')
                plt.draw()
                plt.pause(0.3)

                original_best_search = False
                original = True

                #Ranking best
                if original_best_search == True:

                    top_ind = np.sort(np.argsort(pert_fittness,axis =0)[-self.best:][::-1],axis = 0).flatten()
                    pert_fittness = pert_fittness[top_ind]
                    gradient = (1 / len(top_ind) / self.sigma * (noises[top_ind,:].T@pert_fittness)).flatten()
                    SGD_gradient = SGD_.get_gradients(gradient)
                    param = param + SGD_gradient

                #vanllia
                elif original == True:
                    gradient = (1 / self.num_perturbations / self.sigma * (noises.T@pert_fittness)).flatten()
                    SGD_gradient = SGD_.get_gradients(gradient)
                    # print(SGD_gradient)
                    param = param + SGD_gradient

                #ARS
                else:
                    fb_fittness = np.hstack((pert_fittness,anti_pert_fittness))
                    top_ind = np.sort(np.argsort(np.max(fb_fittness,axis = 1,keepdims = True),axis = 0)[-self.best:][::-1],axis = 0).flatten()
                    # print(top_ind.shape)
                    fit_diff = pert_fittness - anti_pert_fittness
                    reward_noise = np.std(np.vstack((pert_fittness,anti_pert_fittness)))
                    fit_diff = fit_diff[top_ind]
                    # print(fit_diff.shape)
                    # print(noises[top_ind,:].shape)
                    gradient = (1 / len(top_ind) / self.sigma /reward_noise * (noises[top_ind,:].T@fit_diff)).flatten()
                    SGD_gradient = SGD_.get_gradients(gradient)
                    param = param + SGD_gradient

                if iteration % 50 == 0:
                    normal = Normalizer(state_space)
                    bbnp.roll_out(param,env,normal,render = True)
                print("-" * 100)

                #print the results
                print('iteration : {} |  average_fit : {}'.format(iteration,average_fit))


                if average_fit > best_reward:
                   self.save_param(path,name,param,average_fit,iteration)
            #save weights
            # bbnw.saver.save(bbnw.sess,)
            # np.save(param_dir + 'param', jkj)
    
                
    def save_param(self,path,name,param,average_fit,iteration):
        try:
            param_dict['param'].append(param.tolist())
        except:
            param_dict['param'] = []
        try:
            param_dict['avg_fit'].append(average_fit)
        except:
            param_dict['avg_fit'] = []
        try:
            param_dict['iteration'].append(iteration)
        except:
            param_dict['iteration'] = []

        with open(path + name, 'w') as outfile:
            json.dump(param_dict, outfile)




if __name__ == '__main__':
    ES = train_ES(env = 'MountainCarContinuous-v0',continuous_action = True)
    ES.train()

