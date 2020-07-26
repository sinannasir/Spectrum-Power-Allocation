# -*- coding: utf-8 -*-
"""
@author: anonymous
"""

import numpy as np
import project_backend as pb
import json
import argparse


def main(args):
    
    json_file = args.json_file


    num_sim = args.num_sim
    
    with open ('./config/deployment/'+json_file+'.json','r') as f:
        options = json.load(f)
    
    ## Kumber of samples
    total_samples = options['simulation']['total_samples']
        
    N = options['simulation']['N']
    
    # Multi channel scenario, M denotes number of channels.
    if'M' in options['simulation']:
        M = options['simulation']['M']
    else: M = 1
    
    # PFS set to true means that we save log average sum-rate instead of sum-rate
    pfs = False
    if'pfs' in options['simulation']:
        pfs = options['simulation']['pfs']
    
    
    
    # Kow assume each time slot is 1ms and 
    isTrain = options['simulation']['isTrain']
    if isTrain and num_sim == -1:
        num_simulations = options['simulation']['num_simulations']
        simulation = options['simulation']['simulation_index_start']
    elif isTrain:
        num_simulations = 1
        simulation = num_sim
    else:
        simulation = 0
        num_simulations = 1
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
    
    for overal_sims in range(simulation,simulation+num_simulations):
        if isTrain:
            np.random.seed(50+overal_sims)
        else:
            np.random.seed(1050 + overal_sims + N)
        file_path = './simulations/channel/%s_network%d'%(json_file,overal_sims)
        data = np.load(file_path+'.npz',allow_pickle=True)
        
        H_all = data['arr_1']
        
        # Init Optimizer results
        p_FP_nodelay= []
        alpha_FP_nodelay = []
        time_FP_nodelay = []
        
        print('Ideal Case Run sim %d'%(overal_sims))
        print('Run FP sim %d'%(overal_sims))
        ##################### BENCHMARKS #####################
        # In this simulation I assume that the central allocator directly uses the most recent channel condition available.
        # Sum rate
        sum_rate_nodelay = []
        sum_rate_FPMulti_delayedbyone = []
        sum_rate_randomCS_randomP = []
        if not pfs:
            weights = []
            for loop in range(total_samples):
                weights.append(np.array(np.ones(N)))    
            # (p_FP_nodelay,alpha_FP_nodelay,time_FP_nodelay) = zip(*[pb.FP_algorithm_multi(N, M, H, Pmax, noise_var,weight) for (H,weight) in zip(H_all,weights)])
            ii = 0
            for (H,weight) in zip(H_all,weights): 
                aa,bb,cc = pb.FP_algorithm_multi(N, M, H, Pmax, noise_var,weight)
                p_FP_nodelay.append(aa)
                alpha_FP_nodelay.append(bb)
                time_FP_nodelay.append(cc)
                if ii%100 == 0:
                    print(ii)
                ii += 1
                    
        #    # General simulations
            # sum_rate_nodelay = [pb.sumrate_multi_weighted_clipped(H,p,alpha,noise_var,weight) for (H,p,alpha,weight) in zip(H_all,p_FP_nodelay,alpha_FP_nodelay,weights)]  
            # Kow, simulate the process where we use the original FP algorithm
            # Assumption is we ignore the delay at the backhaul network, i.e. there is no delay between the UE and the central controller.
           
            # Initial allocation is just random
            p_central = Pmax * np.random.rand(N)
            # all_alpha_combs = pb.permute_alphas(N,M)
            # alpha_central = all_alpha_combs[np.random.randint(len(all_alpha_combs))]
            alpha_central = pb.random_alpha_full(N,M)
            for sim in range (total_samples):
                sum_rate_nodelay.append(pb.sumrate_multi_weighted_clipped(H_all[sim],p_FP_nodelay[sim],alpha_FP_nodelay[sim],noise_var,weights[sim]))
                if (sim > 0):
                    p_central = p_FP_nodelay[sim-1]
                    alpha_central = alpha_FP_nodelay[sim-1]
                sum_rate_FPMulti_delayedbyone.append(pb.sumrate_multi_weighted_clipped(H_all[sim],p_central,alpha_central,noise_var,weights[sim]))
                
                random_alpha = pb.random_alpha_full(N,M)#all_alpha_combs[np.random.randint(len(all_alpha_combs))]
                # rand_p,_ = pb.FP_algorithm_multi_knownchannel(N,random_alpha, H_all[sim], Pmax, noise_var,weights[sim])
                # sum_rate_randomCS_idealFP.append(pb.sumrate_multi_weighted_clipped(H_all[sim],rand_p,random_alpha,noise_var,weights[sim]))
                
                sum_rate_randomCS_randomP.append(pb.sumrate_multi_weighted_clipped(H_all[sim],Pmax * np.random.rand(N),random_alpha,noise_var,weights[sim]))
        else:
            beta = 0.01
            for sim in range(total_samples):
                if sim % train_episodes['T_train'] == 0: # Restart
                    p_FP_nodelay.append(Pmax*np.ones(N))
                    alpha_FP_nodelay.append(np.zeros((N,M)))
                    alpha_FP_nodelay[-1][:,0] = 1
                    rate = [1e-10+np.array(pb.sumrate_multi_list_clipped(H_all[sim],p_FP_nodelay[-1],alpha_FP_nodelay[-1],noise_var))]
                    average_sum_rate = np.array(rate[-1])
                    weights = [np.array([1.0/i for i in average_sum_rate])]
                    sum_rate_nodelay.append(np.sum(np.log(average_sum_rate)))
                    time_FP_nodelay = [[0,0]]
                else:
                    tmp_FP_p, tmp_FP_alpha, cc = pb.FP_algorithm_multi(N,M, H_all[sim], Pmax, noise_var,weights[-1])
                    p_FP_nodelay.append(tmp_FP_p)
                    alpha_FP_nodelay.append(tmp_FP_alpha)
                    time_FP_nodelay.append(cc)
                    rate.append(pb.sumrate_multi_list_clipped(H_all[sim],tmp_FP_p,tmp_FP_alpha,noise_var))
                    average_sum_rate = (1.0-beta)*average_sum_rate+beta*np.array(rate[-1])
                    sum_rate_nodelay.append(np.sum(np.log(average_sum_rate)))
                    weights.append(np.array([1.0/i for i in average_sum_rate]))
                if(sim%100 == 0):
                    print(sim)
            print('get sum_rate_FPMulti_delayedbyone')
            for sim in range(total_samples):
                if sim % train_episodes['T_train'] == 0: # Restart
                    allone_alpha = np.zeros((N,M))
                    allone_alpha[:,0] = 1
                    rate = [1e-10+np.array(pb.sumrate_multi_list_clipped(H_all[sim],Pmax*np.ones(N),allone_alpha,noise_var))]
                    average_sum_rate = np.array(rate[-1])
                    weights = [np.array([1.0/i for i in average_sum_rate])]
                    sum_rate_FPMulti_delayedbyone.append(np.sum(np.log(average_sum_rate)))
                else:
                    tmp_FP_p, tmp_FP_alpha, cc = pb.FP_algorithm_multi(N,M, H_all[sim-1], Pmax, noise_var,weights[-1])
                    rate.append(pb.sumrate_multi_list_clipped(H_all[sim],tmp_FP_p,tmp_FP_alpha,noise_var))
                    average_sum_rate = (1.0-beta)*average_sum_rate+beta*np.array(rate[-1])
                    sum_rate_FPMulti_delayedbyone.append(np.sum(np.log(average_sum_rate)))
                    weights.append(np.array([1.0/i for i in average_sum_rate]))
                if(sim%100 == 0):
                    print(sim)
            # print('get sum_rate_randomCS_idealFP')
            # for sim in range(total_samples):
            #     if sim % train_episodes['T_train'] == 0: # Restart
            #         allone_alpha = np.zeros((N,M))
            #         allone_alpha[:,0] = 1
            #         rate = [1e-10+np.array(pb.sumrate_multi_list_clipped(H_all[sim],Pmax*np.ones(N),allone_alpha,noise_var))]
            #         average_sum_rate = np.array(rate[-1])
            #         weights = [np.array([1.0/i for i in average_sum_rate])]
            #         sum_rate_randomCS_idealFP.append(np.sum(np.log(average_sum_rate)))
            #     else:
            #         tmp_FP_alpha = pb.random_alpha_full(N,M)
            #         tmp_FP_p,_ = pb.FP_algorithm_multi_knownchannel(N,tmp_FP_alpha, H_all[sim], Pmax, noise_var,weights[-1])
            #         rate.append(pb.sumrate_multi_list_clipped(H_all[sim],tmp_FP_p,tmp_FP_alpha,noise_var))
            #         average_sum_rate = (1.0-beta)*average_sum_rate+beta*np.array(rate[-1])
            #         sum_rate_randomCS_idealFP.append(np.sum(np.log(average_sum_rate)))
            #         weights.append(np.array([1.0/i for i in average_sum_rate]))
            #     if(sim%100 == 0):
            #         print(sim)
            print('get sum_rate_randomCS_randomP')
            for sim in range(total_samples):
                if sim % train_episodes['T_train'] == 0: # Restart
                    allone_alpha = np.zeros((N,M))
                    allone_alpha[:,0] = 1
                    rate = [1e-10+np.array(pb.sumrate_multi_list_clipped(H_all[sim],Pmax*np.ones(N),allone_alpha,noise_var))]
                    average_sum_rate = np.array(rate[-1])
                    weights = [np.array([1.0/i for i in average_sum_rate])]
                    sum_rate_randomCS_randomP.append(np.sum(np.log(average_sum_rate)))
                else:
                    tmp_FP_alpha = pb.random_alpha_full(N,M)
                    tmp_FP_p = Pmax * np.random.rand(N)
                    rate.append(pb.sumrate_multi_list_clipped(H_all[sim],tmp_FP_p,tmp_FP_alpha,noise_var))
                    average_sum_rate = (1.0-beta)*average_sum_rate+beta*np.array(rate[-1])
                    sum_rate_randomCS_randomP.append(np.sum(np.log(average_sum_rate)))
                    weights.append(np.array([1.0/i for i in average_sum_rate]))
                if(sim%100 == 0):
                    print(sim)
        np_save_path = './simulations/sumrate/benchmarks/%s_network%d'%(json_file,overal_sims)
        np.savez(np_save_path,p_FP_nodelay,alpha_FP_nodelay,time_FP_nodelay,sum_rate_nodelay,
                 sum_rate_FPMulti_delayedbyone,sum_rate_randomCS_randomP)
        print('Saved to %s'%(np_save_path))



if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='give test scenarios.')

    parser.add_argument('--json-file', type=str, default='test_K5_N20_M1_shadow10_episode10-500_travel0_fd10',
                       help='json file for the deployment')
    parser.add_argument('--num-sim', type=int, default=0,
                       help='If set to -1, it uses num_simulations of the json file. If set to positive, it runs one simulation with the given id.')
    
    args = parser.parse_args()

    main(args)
    
    
