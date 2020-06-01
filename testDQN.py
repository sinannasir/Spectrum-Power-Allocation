# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""

import numpy as np
import project_backend as pb
import time
import collections
import json
import DQN
import copy
import os

# json_file = "test_N10_K20_shadow10_episode5-2500_travel0_vmax2_5" 
# json_file = "test_N20_K40_shadow10_episode5-2500_travel0_vmax2_5" 
#json_file = "test_N20_K60_shadow10_episode5-2500_travel0_vmax2_5" 
#json_file = "test_N20_K80_shadow10_episode5-2500_travel0_vmax2_5" 
#json_file = "test_N20_K100_shadow10_episode5-2500_travel0_vmax2_5"

#json_file = "test_N30_K60_shadow10_episode5-2500_travel0_vmax2_5" 
#json_file = "test_N40_K80_shadow10_episode5-2500_travel0_vmax2_5" 
#json_file = "test_N50_K100_shadow10_episode5-2500_travel0_vmax2_5" 
#json_file = "test_N100_K200_shadow10_episode5-2500_travel0_vmax2_5" 
#json_file = "test_N150_K300_shadow10_episode5-2500_travel0_vmax2_5" 

#json_file_train = "train_N5_K10_shadow10_episode30-5000_travel0_fd10"
#json_file_train = "train_N5_K10_shadow10_episode30-5000_travel0_fdmixed"
#json_file_train = "train_N5_K10_shadow10_episode30-5000_travel0_vmax2_5"
#json_file_train = "train_N5_K10_shadow10_episode30-5000_travel5000_vmax2_5"
#json_file_train = "train_N5_K10_shadow10_episode30-5000_travel5000_vmax5_0"
#json_file_train = "train_N5_K10_shadow10_episode30-5000_travel20000_vmax2_5"
# json_files_train = ["train_N10_K20_shadow10_episode30-5000_travel0_fd10"]
#json_file_train = "train_N10_K20_shadow10_episode30-5000_travel0_fdmixed"
#json_file_train = "train_N10_K20_shadow10_episode30-5000_travel0_vmax2_5"
#json_file_train = "train_N10_K20_shadow10_episode30-5000_travel5000_vmax2_5"
#json_file_train = "train_N10_K20_shadow10_episode30-5000_travel5000_vmax5_0"
#json_files_train = ["train_N10_K20_shadow10_episode30-5000_travel20000_vmax2_5" ]

#json_files_train = ["train_N5_K10_shadow10_episode30-5000_travel0_fd10",
#                     "train_N5_K10_shadow10_episode30-5000_travel0_fdmixed",
#                     "train_N5_K10_shadow10_episode30-5000_travel0_vmax2_5",
#                     "train_N5_K10_shadow10_episode30-5000_travel5000_vmax2_5",
#                     "train_N5_K10_shadow10_episode30-5000_travel5000_vmax5_0",
#                     "train_N5_K10_shadow10_episode30-5000_travel20000_vmax2_5",
#                     "train_N10_K20_shadow10_episode30-5000_travel0_fd10",
#                     "train_N10_K20_shadow10_episode30-5000_travel0_fdmixed",
#                     "train_N10_K20_shadow10_episode30-5000_travel0_vmax2_5",
#                     "train_N10_K20_shadow10_episode30-5000_travel5000_vmax2_5",
#                     "train_N10_K20_shadow10_episode30-5000_travel5000_vmax5_0",
#                     "train_N10_K20_shadow10_episode30-5000_travel20000_vmax2_5" ]

# json_file_policy_train = 'dqn200_100_50'

import argparse


json_file = "test_K10_N20_shadow10_episode5-2500_travel0_vmax2_5"
json_file = "test_K20_N40_shadow10_episode5-2500_travel0_vmax2_5" 
json_file = "test_K20_N60_shadow10_episode5-2500_travel0_vmax2_5_" 
json_file = "test_K20_N80_shadow10_episode5-2500_travel0_vmax2_5" 
json_file = "test_K20_N100_shadow10_episode5-2500_travel0_vmax2_5" 

json_files_train = ["train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5",
                    "train_K10_N20_shadow10_episode10-5000_travel0_fd10"]
json_files_train = ["train_K10_N20_shadow10_episode2-5000_travel50000_vmax2_5",
                    "train_K10_N20_shadow10_episode2-5000_travel0_fd10"]

json_file_policy_train = 'ddpg200_100_50'

parser = argparse.ArgumentParser(description='give test scenarios.')
parser.add_argument('--json-file', type=str, default='test_K10_N20_shadow10_episode5-2500_travel0_vmax2_5',
                   help='json file for the deployment')
parser.add_argument('--json-files-train', nargs='+', default=["train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5",
                    "train_K10_N20_shadow10_episode10-5000_travel0_fd10"],
                   help='json files train.')
parser.add_argument('--json-file-PA-train', type=str, default='dqn200_200_100',
                   help='json file for the hyperparameters')
parser.add_argument('--json-file-CS-train', type=str, default='dqn200_200_100',
                   help='json file for the hyperparameters')

args = parser.parse_args()

# json_file = "test_K5_N10_M1_shadow10_episode5-1000_travel0_fd5"
# json_files_train = ["train_K5_N10_M1_shadow10_episode1-25000_travel0_fd5"]

# json_file = "test_K5_N10_M1_shadow10_episode5-1000_travel0_fd15"
# json_files_train = ["train_K5_N10_M1_shadow10_episode1-25000_travel0_fd15"]

# json_file = "test_K5_N20_M4_shadow10_episode5-1000_travel0_fd5"
# json_files_train = ["train_K5_N20_M4_shadow10_episode1-25000_travel0_fd5"]

# json_file = "test_K5_N20_M4_shadow10_episode5-1000_travel0_fd15"
# json_files_train = ["train_K5_N20_M4_shadow10_episode1-25000_travel0_fd15"]


# json_file = "test_K10_N50_M1_shadow10_episode10-500_travel0_fd5"
# json_files_train = ["train_K10_N50_M1_shadow10_episode5-5000_travelIND_fd5"]

# json_file = "test_K10_N50_M1_shadow10_episode10-500_travel0_fd15"
# json_files_train = ["train_K10_N50_M1_shadow10_episode5-5000_travelIND_fd15"]

# json_file = "test_K10_N50_M5_shadow10_episode5-1000_travel0_fd5"
# json_files_train = ["train_K10_N50_M5_shadow10_episode5-5000_travelIND_fd5"]

# json_file = "test_K10_N50_M5_shadow10_episode5-1000_travel0_fd15"
# json_files_train = ["train_K10_N50_M5_shadow10_episode5-5000_travelIND_fd15"]


# json_file = "test_K10_N50_M10_shadow10_episode5-1000_travel0_fd5"
# json_files_train = ["train_K10_N50_M10_shadow10_episode5-5000_travelIND_fd5"]


json_file = "test_K5_N20_M1_shadow10_episode10-500_travel0_fd10"
json_files_train = ["train_K5_N20_M1_shadow10_episode5-5000_travelIND_fd10"]

json_file_policy_train = args.json_file_PA_train
json_file_policy_CS_train = args.json_file_CS_train


with open ('./config/deployment/'+json_file+'.json','r') as f:
    options = json.load(f)
with open ('./config/policy/'+json_file_policy_train+'.json','r') as f:
    options_policy = json.load(f)
with open ('./config/policy/'+json_file_policy_CS_train+'.json','r') as f:
    options_CS = json.load(f)
if not options_policy['cuda']:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
for json_file_train in json_files_train:
    with open ('./config/deployment/'+json_file_train+'.json','r') as f:
        options_train = json.load(f)
    included_train_episodes = []
    tot_train_episodes = int(options_train['simulation']['total_samples']/options_train['train_episodes']['T_train'])
    N = options['simulation']['N']
    # Multi channel scenario, M denotes number of channels.
    if'M' in options['simulation']:
        M = options['simulation']['M']
    else: M = 1
    # if N <=20:
    #     for i in range(tot_train_episodes+1):
    #         if i<=15 or i%5==0:
    #             included_train_episodes.append(i)
    # else:
    included_train_episodes.append(tot_train_episodes)
    
    train_tot_simulations = options_train['simulation']['num_simulations']
    tot_test_episodes = int(options['simulation']['total_samples']/options['train_episodes']['T_train'])
    inner_train_networks = [[]]*tot_test_episodes
    for i in range(tot_test_episodes):
        inner_train_networks[i] = 0
        # if options['simulation']['test_include'] == 'all':
        #     inner_train_networks[i] = 0#list(range(train_tot_simulations))
        # else:
        #     inner_train_networks[i] = list(np.random.randint(0,train_tot_simulations,options['simulation']['test_include']))
    ## Kumber of samples
    total_samples = options['simulation']['total_samples']
    
    use_prev_all = False
    
    K = options['simulation']['K']
    N = options['simulation']['N']
    equal_number_for_BS = options['simulation']['equal_number_for_BS']
    R_defined = options['simulation']['R_defined']
    R = (2.0/np.sqrt(3))*R_defined
    min_dist = options['simulation']['min_dist']
    dcor = options['simulation']['dcor']
    shadowing_dev = options['simulation']['shadowing_dev']
    
    # Kow assume each time slot is 1ms and 
    T = options['simulation']['T']
    
    # simulation parameters
    train_episodes = options['train_episodes']
    v_c = options['mobility_params']['v_c'] # speed of light
    f_c = options['mobility_params']['f_c']
    mobility_params = options['mobility_params']
    mobility_params['alpha_angle'] = options['mobility_params']['alpha_angle_rad'] * np.pi #radian/sec
    max_doppler = options['mobility_params']['max_doppler'] # Used only when v_max = 0
    #Some defaults
    R = (2.0/np.sqrt(3))*R_defined
    Pmax_dB = 38.0-30
    Pmax = np.power(10.0,Pmax_dB/10)
    n0_dB = -114.0-30
    noise_var = np.power(10.0,n0_dB/10)
    # Hyper aprameters
    N_neighbors = options_policy['N_neighbors']
    neightresh = noise_var*options_policy['neightresh']
    learning_rate_0 = options_policy['learning_rate_0']
    learning_rate_decay = options_policy['learning_rate_decay']
    learning_rate_min = options_policy['learning_rate_min']
    discount_factor = options_policy['discount_factor']
    scale_R_inner = options_policy['scale_R_inner']
    scale_R_interf = options_policy['scale_R_interf']
    batch_size = options_policy['batch_size']
    memory_per_agent = options_policy['memory_per_agent']
    # epsilon greedy algorithm
    max_epsilon = options_policy['max_epsilon']
    epsilon_decay = options_policy['epsilon_decay']
    min_epsilon = options_policy['min_epsilon']
    # quasi-static target network update
    target_update_count = options_policy['target_update_count']
    time_slot_to_pass_weights = options_policy['time_slot_to_pass_weights'] # 50 slots needed to pass the weights
    n_hidden_1 = options_policy['n_hiddens'][0]
    n_hidden_2 = options_policy['n_hiddens'][1]
    n_hidden_3 = options_policy['n_hiddens'][2]
    
    for ep in included_train_episodes:
        #
        np.random.seed(500 + N + ep)
        file_path = './simulations/channel/%s_network%d'%(json_file,0)
        data = np.load(file_path+'.npz')
        
        H_all = data['arr_1']
        f_d = data['arr_2']
        H_all_2 = []
        for i in range(total_samples):
            H_all_2.append(H_all[i]**2)
        
        weights = []
        for loop in range(total_samples):
            weights.append(np.array(np.ones(N)))
        
        time_calculating_strategy_takes = []
            
        # Virtual neighbor placer
        neighbors_in = collections.deque([],2)
        neighbors = collections.deque([],2)
    
        sims_pos_p = np.zeros(N).astype(int) - 1
    
        policy = DQN.DQN(options,options_policy,N,M,Pmax,noise_var,seed=500 + N + ep)
    
        ## Our JSAC version uses a linear quantizer.
        strategy_translation = np.zeros(policy.power_levels)
        strategy_translation[0] = 0.0 # Tx power 0
        # Calculate steps in dBm
        strategy_translation_dB_step = Pmax_dB/(policy.power_levels-1)
        for i in range(1,policy.power_levels-1):
            strategy_translation[i] = i *(Pmax/(policy.power_levels-1))
        strategy_translation[-1] = Pmax
       
        time_calculating_strategy_takes = []
        time_optimization_at_each_slot_takes = []
        sum_rate_distributed_policy_episode = []
        p_strategy_all_apisode = []
        i_train = 0
#        for i_train in range(len(inner_train_networks[0])):
        sum_rate_distributed_policy = []
        sum_rate_list_distributed_policy = collections.deque([],2)
        # Initial allocation is just random
        p_central = Pmax * np.random.rand(N)
        p_strategy = np.array(p_central) # strategy is a completely different object
        p_strategy_current = np.array(p_strategy)
        
        alpha_central = np.zeros((N,M))
        for k in range(N):
            alpha_central[k,np.random.randint(M)] = 1
        alpha_strategy = np.array(alpha_central) # strategy is a completely different object
        alpha_strategy_current = np.array(alpha_strategy)
        
        alpha_int_central = np.where(alpha_central==1)[1].astype(int)
        alpha_int_strategy = np.array(alpha_central) # strategy is a completely different object
        alpha_int_strategy_current = np.array(alpha_int_strategy)
       
        # current CSI used to calculate the power allocation
        current_csi = 0
        previous_csi = 0
       
        
        p_strategy_all=[]
        alpha_strategy_all = []
        alpha_int_strategy_all = []
    
        with tf.Session() as sess:
            sess.run(policy.init)
            policy.initialize_updates(sess) 
            # Start iterating voer time slots
            for sim in range (total_samples):
                # save an instance per training episode for testing purposes.
                if(sim %train_episodes['T_train'] == 0):
                    train_network_idx = i_train#inner_train_networks[int(sim /train_episodes['T_train'])][i_train]
                    model_destination = ('./simulations/sumrate/policy/%s_%s_%s_network%d_episode%d.ckpt'%(
                            json_file_train,json_file_policy_train,json_file_policy_CS_train,train_network_idx,ep)).replace('[','').replace(']','')
                    policy.load(sess,model_destination)
                    i_train+=1
                    i_train = i_train % train_tot_simulations
        
                # If at least one time slot passed to get experience
                if (sim %train_episodes['T_train'] > 1):                    
                    # Each agent picks its strategy.
                    for agent in range (N):
                        current_local_state = policy.local_state(sim,agent,p_strategy_all,alpha_strategy_all,H_all_2,sum_rate_list_distributed_policy,weights) 
                        a_time = time.time()  
                        strategy = policy.act_noepsilon(sess,current_local_state,sim)
                        time_calculating_strategy_takes.append(time.time()-a_time)
                                
                        # Pick the action
                        p_strategy[agent] = strategy_translation[strategy % policy.power_levels]
                        alpha_strategy[agent,:] = np.zeros(M)
                        alpha_strategy[agent,strategy // policy.power_levels] = 1
                        alpha_int_strategy[agent] = strategy // policy.power_levels
    
                        # Add current state to the short term memory to observe it during the next state
                        policy.previous_state[agent,:] = current_local_state
                        policy.previous_action[agent] = strategy
    
                if(sim %train_episodes['T_train'] < 2):
                    p_strategy = np.random.rand(N)
                    alpha_strategy = np.zeros((N,M))
                    for k in range(N):
                        alpha_strategy[k,np.random.randint(M)] = 1
                    alpha_int_strategy = np.where(alpha_strategy==1)[1].astype(int)
                    
                p_strategy_current = np.array(p_strategy)
                alpha_strategy_current = np.array(alpha_strategy)
                alpha_int_strategy_current = np.array(alpha_int_strategy).astype(int)
                for m in range(M):
                    policy.prev_suminterferences[:,m] = np.matmul(H_all_2[sim][:,:,m],alpha_strategy[:,m]*p_strategy) - (H_all_2[sim][:,:,m].diagonal()*alpha_strategy[:,m]*p_strategy) + noise_var
                # sims_pos_p[np.where(p_strategy_current>0)] = sim
    

                sum_rate_list_distributed_policy.append(pb.reward_helper(H_all[sim],p_strategy,alpha_strategy,noise_var,Pmax))

                weights.append(np.array(np.ones(N)))
                sum_rate_distributed_policy.append(pb.sumrate_multi_weighted_clipped(H_all[sim],p_strategy,alpha_strategy,noise_var,weights[sim]))

    
                p_strategy_all.append(p_strategy_current)
                alpha_strategy_all.append(alpha_strategy_current)
                alpha_int_strategy_all.append(alpha_int_strategy_current)
                if(sim%2500 == 0):
                    print('Test time %d'%(sim))
        sum_rate_distributed_policy_episode.append(copy.copy(sum_rate_distributed_policy))
        p_strategy_all_apisode.append(copy.copy(p_strategy_all))
               
        # End Train Phase
        np_save_path = './simulations/sumrate/test/%s_%s_%s_%s_episode%d.ckpt'%(json_file,json_file_train,json_file_policy_train,json_file_policy_CS_train,ep)
        print(np_save_path)
        np.savez(np_save_path,options,options_policy,sum_rate_distributed_policy_episode,p_strategy_all_apisode,
                 time_optimization_at_each_slot_takes,time_calculating_strategy_takes,included_train_episodes,inner_train_networks)
    
