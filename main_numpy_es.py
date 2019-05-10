from bb_numpy_net import bb_numpy_swimmer as bb_numpy
import multiprocessing as mp
from multiprocessing import Queue, Process
import os
import matplotlib.pyplot as plt
import gym
import numpy as np
from scipy.linalg import hadamard
from scipy.spatial.distance import pdist, squareform, cdist
from utils import SGD, get_info_summary,get_noise_matrices, compute_weight_decay,sample, gaussian_kernelize,get_sympoly,cond_kdpp
from Normalizer import Normalizer
import json
import time
import sys
import cma

class train_ES():
    def __init__(
                self,
                iterations = 20000,
                num_perturbations = 64,
                env = 'BipedalWalker-v2',
#                env = 'CartPole-v1',
#                env = 'MountainCarContinuous-v0',    # depending on how it is initialized - could possibly fall into local minimum
#                env = 'MountainCar-v0',
                # Swimmer-v2
                gamma = 0.99,
                sigma = 0.01,
                lr = 3 * 1e-2,
                max_length = 2000,
                num_test = 1,
                continuous_action = False,
                best = 64,
                reward_renormalize = False,
                state_renormalize = False,
                video_save_interval = 100,
                num_cpu = 8,
                overlap = 0.2,
                noise_type = 'Gaussian',
                method_type = 'Vanilla',
                test = 0,
                perc_reuse = .2
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
        self.noise_type = noise_type
        self.method_type = method_type
        self.test = test
        self.video_save_interval = video_save_interval
        self.perc_reuse = perc_reuse
        # self.cond_DPP = cond_DPP
        self.buffer = []

        self.k = self.num_perturbations - int(self.num_perturbations * overlap)

        global param_dir 
        param_dir = 'Param'
        global path 
        path = param_dir + '/' + env + '_' + str(self.method_type) + '_perturbations_' + str(self.num_perturbations) + '_' +'state_renormalize_' + str(self.state_renormalize)+ '_Sigma_' + str(self.sigma) + "_best_" + str(self.best) + "_test_" +  str(self.test)
        global name
        name = 'data.json'
        global param_dict
        param_dict = {}


    def _do_work(self,queue,bbnp,param,action_space,state_space,max_length,seed_id,num_workers,env,continuous_action,cma_param,index_start):
        start = time.time()
        np.random.seed(seed_id)
        
        if self.noise_type == 'Gaussian':
            noises = self.sigma * np.random.randn(num_workers,len(param))
            noisy_param = param + noises
        elif self.noise_type == 'Hadamard':
            h_size = 1<<((max(num_workers,len(param))-1).bit_length())
            h = hadamard(h_size)
            noises = self.sigma*(h@np.diag(np.random.choice([-1,1], h_size)))[:num_workers,:len(param)]
            noisy_param = param + noises
        elif self.noise_type == 'CMA':
            noisy_param = cma_param
        elif self.noise_type == 'CDPP':
            noisy_param = self.buffer[index_start:index_start+num_workers]
        
        fitness = []
        anti_fitness = []
        worker_summary = {}

        # fitness_idx_dict = {}
        # anti_fitness_idx_dict = {}
        # idx  = 0


        for ind in noisy_param:

            #do the roll out
            if self.state_renormalize == True:
                normal = Normalizer(state_space)
                ind_fit = bbnp.roll_out(ind,env,self.env,normal,render = False,state_renormalize = True)
                normal = Normalizer(state_space)
                ind_fit_anti = bbnp.roll_out(-ind,env,self.env,normal,render = False,state_renormalize = True)
            else:
                normal = Normalizer(state_space)
                ind_fit = bbnp.roll_out(ind,env,self.env,normal,render = False,init = False)
                ind_fit_anti = bbnp.roll_out(-ind,env,self.env,normal,render = False,init = False)


            fitness.append(ind_fit)
            anti_fitness.append(ind_fit_anti)

        end = time.time()
        
        # print('process id:{}  |  queue time:{}  |  seed_id:{}'.format(os.getpid(),end-start,seed_id))

        worker_summary['fit'] = fitness
        worker_summary['anti_fit'] = anti_fitness
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


            if self.continuous_action == True:
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
            
            cma_es = cma.CMAEvolutionStrategy(param,self.sigma,{'popsize': self.num_perturbations,})



            for iteration in range(self.iterations):
                
                ts = time.time()
                
                # if self.num_perturbations % self.num_cpu != 0:
                #     seed_id =  np.random.randint(np.iinfo(np.int32(10)).max, size=self.num_cpu + 1)
                # else:
                #     seed_id = np.random.randint(np.iinfo(np.int32(10)).max, size=self.num_cpu)

                
                cma_param = np.array(cma_es.ask())


                if self.method_type == 'CDPP':
                    if iteration == 0:

                        X = np.random.randn(self.num_perturbations*3,len(param))
                        # print('X.shape=',X.shape)
                        cond_indices = cond_kdpp([], X, k = self.num_perturbations)
                        # print('cond_indices=',cond_indices)
                        self.buffer = X[cond_indices]

                    else:
                        # print(self.buffer)
                        dists = np.linalg.norm(self.buffer-param,axis=1)
                       # print('dists.shape=',dists.shape)
                        num_closest= int(self.perc_reuse*self.num_perturbations)

                        closest_indices = dists.argsort()[:num_closest]
                        reused_samples = self.buffer[closest_indices]
                        X = np.random.randn(self.num_perturbations*3,len(param)) + param
                        cond_indices = cond_kdpp(self.buffer[closest_indices],X,k=(self.num_perturbations-num_closest))

                        self.buffer = np.vstack((self.buffer[closest_indices],X[cond_indices]))
                        # print(self.buffer)

                        all_indices = []





                queue = Queue()
                num_workers = []
                num_perts = self.num_perturbations
                shared_amt = self.num_perturbations//self.num_cpu
                if self.noise_type == 'CDPP' and iteration!=0:
                    num_perts = self.num_perturbations - int(self.perc_reuse*self.num_perturbations)
                    shared_amt = (self.num_perturbations - int(self.perc_reuse*self.num_perturbations))//self.num_cpu
                    # num_workers = [int((self.num_perturbations - int(self.perc_reuse*self.num_perturbations))/ self.num_cpu)] * self.num_cpu + [self.num_perturbations % self.num_cpu]
                while num_perts > shared_amt:
                        num_perts -= shared_amt
                        num_workers.append(shared_amt)
                num_workers.append(num_perts)
                start_indices = [0]
                # print('num_workers=',num_workers)
                for i in range(1,len(num_workers)):
                   start_indices.append(start_indices[i-1]+num_workers[i-1])

                cma_param_slicer = [0]
                cma_param_slicer.extend(num_workers)
                cma_param_slicer = np.cumsum(cma_param_slicer)
                
                seed_id =  np.random.randint(np.iinfo(np.int32(10)).max, size=len(num_workers))


                workers = [Process(target = self._do_work,args = (queue,bbnp,param,action_space,state_space,self.max_length,seed_id[i],num_workers[i],env,self.continuous_action,cma_param[cma_param_slicer[i]:cma_param_slicer[i+1],:],start_indices[i])) for i in range(len(seed_id))]


                for worker in workers:
                    worker.start()

                results = [queue.get() for p in workers]
                
                # Swapping this with the above line so deadlock is avoided
                for worker in workers:
                    worker.join()
                if self.noise_type == 'CDPP':
                    if iteration != 0:
                        # print('pert_fitness.shape=',pert_fitness.shape)
                        # print('closest_indices=',closest_indices) 
                        old_pert_fitness = pert_fitness[closest_indices]
                        old_anti_pert_fitness = anti_pert_fitness[closest_indices]
                
                pert_fitness,anti_pert_fitness,seed_id = get_info_summary(results)
#                print('pert_fit',pert_fitness,'\n','anti_pert_fit',anti_pert_fitness,'\n','seed_id',seed_id)
                pert_fitness = np.array(pert_fitness)[...,None]
                anti_pert_fitness = np.array(anti_pert_fitness)[...,None]

                if self.noise_type == 'CDPP':
                    if iteration != 0:
                        # print('(when stacking) pert_fitness=',pert_fitness.shape)
                        # print('(when stacking) old_pert_fitness=',old_pert_fitness.shape)
                        pert_fitness = np.vstack((old_pert_fitness,pert_fitness))
                        anti_pert_fitness = np.vstack((old_anti_pert_fitness,anti_pert_fitness))


                if self.noise_type in ['Gaussian','Hadamard']:
                    noises = get_noise_matrices(seed_id,num_workers,len(param),self.sigma,self.noise_type)

                #record average_fit
                average_fit = np.sum(pert_fitness)/self.num_perturbations

                fit_list.append(average_fit)
                iteration_list.append(iteration)

                #dynamic plot the graph
                plt.plot(iteration_list,fit_list,'r')
                plt.draw()
                plt.pause(0.3)

                #Ranking best
                if self.method_type == 'Rank':

                    top_ind = np.sort(np.argsort(pert_fitness,axis =0)[-self.best:][::-1],axis = 0).flatten()
                    pert_fitness = pert_fitness[top_ind]
                    gradient = (1 / len(top_ind) / self.sigma * (noises[top_ind,:].T@pert_fitness)).flatten()
                    SGD_gradient = SGD_.get_gradients(gradient)
                    param = param + SGD_gradient
                    print('param',param)


                #Vanilla
                elif self.method_type == 'Vanilla':
                    gradient = (1 / self.num_perturbations / self.sigma * (noises.T@pert_fitness)).flatten()
                    SGD_gradient = SGD_.get_gradients(gradient)
                    param = param + SGD_gradient
                    
                #CMA
                elif self.method_type == 'CMA':
                    cma_es.tell(cma_param,-pert_fitness[:,0] - compute_weight_decay(0.01,cma_param))
                    param = cma_es.result[5] # mean of all perturbations - for render and save - not used to update new space

                #ARS
                elif self.method_type == 'ARS':
                    # fb_fitness = np.hstack((pert_fitness,anti_pert_fitness))
                    # top_ind = np.sort(np.argsort(np.max(fb_fitness,axis = 1,keepdims = True),axis = 0)[-self.best:][::-1],axis = 0).flatten()
                    # # print(top_ind.shape)
                    # fit_diff = pert_fitness - anti_pert_fitness
                    # reward_noise = np.std(np.vstack((pert_fitness,anti_pert_fitness)))
                    # fit_diff = fit_diff[top_ind]
                    # # print(fit_diff.shape)
                    # # print(noises[top_ind,:].shape)
                    # gradient = (1 / len(top_ind) / self.sigma /reward_noise * (noises[top_ind,:].T@fit_diff)).flatten()
                    # SGD_gradient = SGD_.get_gradients(gradient)
                    # param = param + SGD_gradient

                    fb_fitness = np.hstack((pert_fitness,anti_pert_fitness))
                    top_ind = (np.argsort(np.max(fb_fitness,axis = 1,keepdims = True),axis = 0)[-self.best:][::-1]).flatten()
                    # print(top_ind.shape)
                    fit_diff = pert_fitness - anti_pert_fitness
                    reward_noise1 = np.std(np.vstack((pert_fitness,anti_pert_fitness)))
                    fit_diff = fit_diff[top_ind]

                    reward_noise2 = np.std(np.max(fb_fitness[top_ind],axis = 0),axis = 0)
                    reward_noise3 = np.std(fit_diff,axis = 0)
                    # print(fit_diff.shape)
                    # print(noises[top_ind,:].shape)
                    gradient = (1 / len(top_ind) / self.sigma/reward_noise1 * (noises[top_ind,:].T@fit_diff)).flatten()
                    SGD_gradient = SGD_.get_gradients(gradient)
                    param = param + SGD_gradient



                #CDPP
                elif self.method_type == 'CDPP':
                    # print('self.buffer.shape = ',self.buffer.shape)
                    # print('param.shape = ',param.shape)
                    cond_noise = self.buffer - param
                    if iteration != 0:
                    #     print('cond_noise.shape=',cond_noise.shape,'reused_samples.shape=',reused_samples.shape)
                        noises = np.vstack((reused_samples,cond_noise))
                    else:
                        noises = cond_noise


                    top_ind = np.sort(np.argsort(pert_fitness,axis =0)[-self.best:][::-1],axis = 0).flatten()
                    best_pert_fitness = pert_fitness[top_ind]
                    # gradient = (1 / len(top_ind) / self.sigma * (noises[ctop_ind,:].T@pert_fitness)).flatten()
                    gradient = (1 / self.num_perturbations / self.sigma * ((self.buffer - param)[top_ind,:].T@best_pert_fitness)).flatten()
                    # SGD_gradient = SGD_.get_gradients(gradient)
#                    print("gradient")
#                    print(SGD_gradient)
#                    print("param")
#                    print(param)
                    # param = param + SGD_gradient
                    param = param + self.lr * gradient
                    # print('param',param)

                if iteration % self.video_save_interval == 0 and iteration !=0:
                    normal = Normalizer(state_space)
                    video_env = gym.wrappers.Monitor(env, './videos/' + str(self.env) + '/'+ str(self.method_type) + '_perturbations_' + str(self.num_perturbations) + '_' +'state_renormalize_' + str(self.state_renormalize)+ '_Simga_' + str(self.sigma) + "_best_" + str(self.best) + "_iter_" +str(iteration) + "_test_" + str(self.test))
                    bbnp.roll_out(param,video_env,self.env,normal,render = True)
                print("-" * 100)

                te = time.time()

                #print the results
                print('iteration: {} | average_fit: {} | # params: {} | time: {:2.2f}s'.format(iteration,average_fit,len(param),te-ts))


                # if average_fit > best_reward:
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


    # for i in range(5):
    #     ES = train_ES(env = 'MountainCarContinuous-v0',continuous_action = True,noise_type='Gaussian',method_type='Vanilla',state_renormalize = False,video_save_interval = 200,iterations = 201)
    #     ES.train()

    # for i in range(5):
    #     ES = train_ES(env = 'MountainCarContinuous-v0',sigma = 0.3,continuous_action = False,noise_type='Gaussian',method_type='ARS',state_renormalize = True,video_save_interval = 200,test = i,iterations = 201)
    #     ES.train()

    # for i in range(5):
    #     ES = train_ES(env = 'MountainCarContinuous-v0',sigma = 0.3,continuous_action = True,noise_type='Gaussian',method_type='Rank',state_renormalize = True,video_save_interval = 200,test = i,iterations = 201)
    #     ES.train()

    # for i in range(5):
    #     ES = train_ES(env = 'MountainCarContinuous-v0',sigma = 1,continuous_action = True,noise_type='CMA',method_type='CMA',state_renormalize = False,num_perturbations = 128,video_save_interval = 200,test = i,iterations = 201)
    #     ES.train()

    # for i in range(5):
    #     ES = train_ES(env = 'MountainCarContinuous-v0',sigma = 0.3,continuous_action = True,noise_type='CDPP',method_type='CDPP',state_renormalize = False,video_save_interval = 200,test = i,iterations = 201)
    #     ES.train()


    for i in range(5):
        ES = train_ES(env = 'Swimmer-v2',sigma = 0.3,continuous_action = True,noise_type='Gaussian',method_type='Vanilla',state_renormalize = True,video_save_interval = 200,test = i,iterations = 201)
        ES.train()

    # for i in range(5):
    #     ES = train_ES(env = 'Swimmer-v2',sigma = 0.1,continuous_action = True,noise_type='Gaussian',method_type='ARS',state_renormalize = True,video_save_interval = 200,test = i,iterations = 201)
    #     ES.train()

    # for i in range(5):
    #     ES = train_ES(env = 'Swimmer-v2',sigma = 0.3,continuous_action = True,noise_type='Gaussian',method_type='Rank',state_renormalize = False,video_save_interval = 200,test = i,iterations = 201,best=32)
    #     ES.train()

    # for i in range(5):
    #     ES = train_ES(env = 'Swimmer-v2',sigma = 1,continuous_action = True,noise_type='CMA',method_type='CMA',state_renormalize = False,video_save_interval = 200,test = i,iterations = 201)
    #     ES.train()

    # for i in range(5):
    #     ES = train_ES(env = 'Swimmer-v2',sigma = 0.3,continuous_action = True,noise_type='CDPP',method_type='CDPP',state_renormalize = False,video_save_interval = 200,test = i,iterations = 201)
    #     ES.train()
