# -*- coding: utf-8 -*-
"""
@author: anonymous
"""

import numpy as np
import project_backend as pb
import time
import collections
import json
import DDPG
import copy
import os

import argparse




def main(args):
    
    json_file = args.json_file
    json_files_train = args.json_files_train
        
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
        
        
        
        # simulation parameters
        train_episodes = options['train_episodes']
        mobility_params = options['mobility_params']
        mobility_params['alpha_angle'] = options['mobility_params']['alpha_angle_rad'] * np.pi #radian/sec
        #Some defaults
        Pmax_dB = 38.0-30
        Pmax = np.power(10.0,Pmax_dB/10)
        n0_dB = -114.0-30
        noise_var = np.power(10.0,n0_dB/10)
        # Hyper aprameters
        neightresh = noise_var*options_policy['neightresh']        
        i_train = -1
        
        for ep in included_train_episodes:
            #
            np.random.seed(500 + N + ep)
            # i_train = np.random.randint(train_tot_simulations)
            i_train+=1
            i_train = i_train % train_tot_simulations
            
            file_path = './simulations/channel/%s_network%d'%(json_file,0)
            data = np.load(file_path+'.npz')
            
            H_all = data['arr_1']
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
            policy = DDPG.DDPG(options,options_policy,options_CS,N,M,Pmax,noise_var, seed=500 + N + ep)
           
            time_calculating_strategy_takes = []
            time_optimization_at_each_slot_takes = []
            sum_rate_distributed_policy_episode = []
            p_strategy_all_episode = []
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
                      
            p_strategy_all=[]
            alpha_strategy_all = []
            alpha_int_strategy_all = []
            with tf.Session() as sess:
                sess.run(policy.init)
                policy.initialize_critic_updates(sess) 
                policy.initialize_actor_updates(sess) 
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
                            # Channel Selection #               
                            current_local_state = policy.local_state(sim,agent,p_strategy_all,alpha_strategy_all,alpha_int_strategy_all,H_all_2,sum_rate_list_distributed_policy,weights) 
                            a_time = time.time()  
                            CSstrategy = policy.CSact_noepsilon(sess,current_local_state,sim)
                            selected_channel = int(CSstrategy)
                            current_singlechannel_state = current_local_state[selected_channel*policy.DDPGnum_input:(selected_channel+1)*policy.DDPGnum_input]
                            # if sim > 1000 and forcezero:
                            #     print('aaa')
                            PAstrategy = policy.PAact_noepsilon(sess,current_singlechannel_state,sim)
                            time_calculating_strategy_takes.append(time.time()-a_time)
                            # if sim == 200:
                            #     print('debug')
                            
                            # Pick the action
                            p_strategy[agent] = policy.Pmax * PAstrategy #** 10
                            # p_strategy[agent] = policy.Pmax * np.round(PAstrategy,2) #** 10
                            alpha_strategy[agent,:] = np.zeros(M)
                            alpha_strategy[agent,CSstrategy] = 1
                            alpha_int_strategy[agent] = selected_channel
        
                            # Add current state to the short term memory to observe it during the next state
                            policy.previous_state[agent,:] = current_singlechannel_state
                            policy.previous_action[agent] = PAstrategy
                            policy.DQNprevious_state[agent,:] = current_local_state
                            policy.DQNprevious_action[agent] = CSstrategy
        
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
                    if M > 1:
                        policy.sorted_channels = np.argsort(H_all_2[sim][np.arange(N),np.arange(N),:]/policy.prev_suminterferences)/float(M)
                    # sims_pos_p[np.where(p_strategy_current>0)] = sim
        
                    # tmp_neighbors_in = []
                    # tmp_neighbors = []
                    # for nei_i in range(N):
                    #     neigh_tmp_variab = np.where((H_all[sim][nei_i,:]**2)*p_strategy_current>neightresh)
                    #     neigh_tmp_variab = np.delete(neigh_tmp_variab,np.where(neigh_tmp_variab[0]==nei_i))
                    #     tmp_neighbors_in.append(neigh_tmp_variab)
        
                    # for nei_i in range(N):
                    #     tmp_neighlist = []
                    #     for nei_j in range(N):
                    #         if(len(np.where(tmp_neighbors_in[nei_j]==nei_i)[0]) != 0):
                    #             tmp_neighlist.append(nei_j)
                    #     if (len(tmp_neighlist) == 0 and len(neighbors) >0):
                    #         tmp_neighbors.append(np.array(neighbors[-1][nei_i]))
                    #     else:
                    #         tmp_neighbors.append(np.array(tmp_neighlist))
                    # neighbors.append(tmp_neighbors)
                    # neighbors_in.append(tmp_neighbors_in)
                    # all sumrates in a list
                    sum_rate_list_distributed_policy.append(pb.reward_helper(H_all[sim],p_strategy,alpha_strategy,noise_var,Pmax))
                    weights.append(np.array(np.ones(N)))
                    sum_rate_distributed_policy.append(pb.sumrate_multi_weighted_clipped(H_all[sim],p_strategy,alpha_strategy,noise_var,weights[sim]))
        
                    p_strategy_all.append(p_strategy_current)
                    alpha_strategy_all.append(alpha_strategy_current)
                    alpha_int_strategy_all.append(alpha_int_strategy_current)
                    
                    if(sim%2500 == 0):
                        print('Test time %d'%(sim))
    
                sum_rate_distributed_policy_episode.append(copy.copy(sum_rate_distributed_policy))
                p_strategy_all_episode.append(copy.copy(p_strategy_all))
            # End Train Phase
            np_save_path = './simulations/sumrate/test/%s_%s_%s_%s_episode%d.ckpt'%(json_file,json_file_train,json_file_policy_train,json_file_policy_CS_train,ep)
            print('Saved to %s'%(np_save_path))
            np.savez(np_save_path,options,options_policy,sum_rate_distributed_policy_episode,p_strategy_all_episode,
                     time_optimization_at_each_slot_takes,time_calculating_strategy_takes,included_train_episodes,inner_train_networks)
    
if __name__ == "__main__":     
    parser = argparse.ArgumentParser(description='give test scenarios.')
    parser.add_argument('--json-file', type=str, default='test_K5_N20_M1_shadow10_episode10-500_travel0_fd10',
                       help='json file for the deployment')
    parser.add_argument('--json-files-train', nargs='+', default=["train_K5_N20_M1_shadow10_episode4-5000_travelIND_fd10"],
                       help='json files train.')
    parser.add_argument('--json-file-PA-train', type=str, default='ddpg200_100_50',
                       help='json file for the hyperparameters')
    parser.add_argument('--json-file-CS-train', type=str, default='dqn100_50_50',
                       help='json file for the hyperparameters')
    
    args = parser.parse_args()
    
    main(args)
    