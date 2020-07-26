# -*- coding: utf-8 -*-
"""
@author: anonymous
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import json
import matplotlib
#matplotlib.use('Qt5Agg')
import argparse




def main(scenario):    
    json_file = scenario['json_file']
    
    
    
    
    json_file_policy = scenario['json_file_policy']
    json_file_CS = scenario['json_file_CS']
    
    json_file_policy2 = scenario['json_file_policy2']
    json_file_CS2 = scenario['json_file_CS2']
    
    num_sim = scenario['num_sim']
    with open ('./config/deployment/'+json_file+'.json','r') as f:
        options = json.load(f)
    
    ## Kumber of samples
    total_samples = options['simulation']['total_samples']
        
    K = options['simulation']['K']
    N = options['simulation']['N']
    # PFS set to true means that we save log average sum-rate instead of sum-rate
    pfs = False
    if'pfs' in options['simulation']:
        pfs = options['simulation']['pfs']
        beta = 0.01
    
    if num_sim == -1:
        num_simulations = options['simulation']['num_simulations']
        simulation = options['simulation']['simulation_index_start']
    else:
        num_simulations = 1
        simulation = num_sim
    
    # simulation parameters
    mobility_params = options['mobility_params']
    mobility_params['alpha_angle'] = options['mobility_params']['alpha_angle_rad'] * np.pi #radian/sec
    
    history = 250
        
    
    mean_p_FP = np.zeros(total_samples)
    mean_time_FP = np.zeros(total_samples)
    mean_iterations_FP = np.zeros(total_samples)
    mean_sum_rate_FP = np.zeros(total_samples)
    
    mean_sum_rate_FPMulti_delayedbyone = np.zeros(total_samples)
    mean_sum_rate_randomCS_idealFP = np.zeros(total_samples)
    mean_sum_rate_randomCS_randomP = np.zeros(total_samples)
    
    mean_sum_rate_policy_train_innersims = np.zeros(total_samples)
    mean_p_strategy_all_train_innersims = np.zeros(total_samples)
    
    mean_time_optimization_at_each_slot_takes = []
    mean_time_calculating_strategy_takes = []
    
    
    mean_sum_rate_policy_train_innersims2 = np.zeros(total_samples)
    mean_p_strategy_all_train_innersims2 = np.zeros(total_samples)
    
    mean_time_optimization_at_each_slot_takes2 = []
    mean_time_calculating_strategy_takes2 = []
    
    for overal_sims in range(simulation,simulation+num_simulations):
    
                 
        # Get the benchmarks.
        file_path = './simulations/sumrate/benchmarks/%s_network%d'%(json_file,overal_sims)
        data = np.load(file_path+'.npz')
        p_FP            = data['arr_0']
        alpha_FP        = data['arr_1']
        time_stats_FP   = data['arr_2']
        sum_rate_FP     = data['arr_3']
        sum_rate_FPMulti_delayedbyone= data['arr_4']
        sum_rate_randomCS_idealFP    = data['arr_5']
        sum_rate_randomCS_randomP    = data['arr_5']
        
        file_path = './simulations/sumrate/train/%s_%s_%s_network%d.ckpt'%(json_file,json_file_policy,json_file_CS,overal_sims)
        data = np.load(file_path+'.npz')
        # Get the train policy results
        sum_rate_policy_train                      = data['arr_2']
        p_strategy_all                          = data['arr_3']
        alpha_strategy_all                      = data['arr_4']
        time_optimization_at_each_slot_takes    = data['arr_5']
        time_calculating_strategy_takes         = data['arr_6']
    
        # Average
        mean_p_FP = mean_p_FP + np.sum(p_FP,1)/float(num_simulations)
        mean_time_FP = mean_time_FP + time_stats_FP[:,0]/float(num_simulations)
        mean_iterations_FP = mean_iterations_FP + time_stats_FP[:,1]/float(num_simulations)
        mean_sum_rate_FP = mean_sum_rate_FP + sum_rate_FP/float(num_simulations)
        
        mean_sum_rate_FPMulti_delayedbyone = mean_sum_rate_FPMulti_delayedbyone + sum_rate_FPMulti_delayedbyone/float(num_simulations)
        mean_sum_rate_randomCS_idealFP = mean_sum_rate_randomCS_idealFP + sum_rate_randomCS_idealFP/float(num_simulations)
        mean_sum_rate_randomCS_randomP = mean_sum_rate_randomCS_randomP + sum_rate_randomCS_randomP/float(num_simulations)
        
        mean_sum_rate_policy_train_innersims = mean_sum_rate_policy_train_innersims + sum_rate_policy_train/float(num_simulations)
        mean_p_strategy_all_train_innersims = mean_p_strategy_all_train_innersims + np.sum(p_strategy_all,1)/float(num_simulations)
        
        mean_time_optimization_at_each_slot_takes.append(time_optimization_at_each_slot_takes)
        mean_time_calculating_strategy_takes.append(time_calculating_strategy_takes)
        
        file_path = './simulations/sumrate/train/%s_%s_%s_network%d.ckpt'%(json_file,json_file_policy2,json_file_CS2,overal_sims)
        data = np.load(file_path+'.npz')
        # Get the train policy results
        sum_rate_policy_train2                      = data['arr_2']
        p_strategy_all2                          = data['arr_3']
        alpha_strategy_all2                      = data['arr_4']
        time_optimization_at_each_slot_takes2    = data['arr_5']
        time_calculating_strategy_takes2         = data['arr_6']
        
        mean_sum_rate_policy_train_innersims2 = mean_sum_rate_policy_train_innersims2 + sum_rate_policy_train2/float(num_simulations)
        mean_p_strategy_all_train_innersims2 = mean_p_strategy_all_train_innersims2 + np.sum(p_strategy_all2,1)/float(num_simulations)
        
        mean_time_optimization_at_each_slot_takes2.append(time_optimization_at_each_slot_takes2)
        mean_time_calculating_strategy_takes2.append(time_calculating_strategy_takes2)
        
    
    if pfs:
        bw = 1e7
        add_bw = np.log(bw)
        
        mean_sum_rate_FP = add_bw + mean_sum_rate_FP 
        mean_sum_rate_FPMulti_delayedbyone = add_bw + mean_sum_rate_FPMulti_delayedbyone
        mean_sum_rate_randomCS_idealFP = add_bw + mean_sum_rate_randomCS_idealFP
        mean_sum_rate_randomCS_randomP = add_bw + mean_sum_rate_randomCS_randomP
        mean_sum_rate_policy_train_innersims = add_bw + mean_sum_rate_policy_train_innersims
        avg_result_over = 1
    else:
        avg_result_over = float(N)
    #print('K '+ str(int(N))+' R '+str(R_defined)+ ' r '+str(min_dist) + ' '+file_path[14:18])
    #print('Test Sum rate optimal ' + str(np.mean(mean_sum_rate[total_samples-2500:]/N)))
    #print('Test Sum rate delayed ' + str(np.mean(mean_sum_rate_FPMulti_delayedbyone[total_samples-2500:]/N)))
    #print('Test Sum rate random ' + str(np.mean(mean_sum_rate_randomCS_idealFP[total_samples-2500:]/N)))
    #print('Test Sum rate max ' + str(np.mean(mean_sum_rate_randomCS_randomP[total_samples-2500:]/N)))
    #for i in range(len(power_multiplier_allsims)):
    #    print('Multiplier '+str(power_multiplier_allsims[i])+
    #          ' Test Sum rate ' +str(np.mean(mean_sum_rate_policy_train_innersims[i,total_samples-2500:]/N)))
    
    
    
    lines = ["-","--",':','-.',':','-.']
    linecycler = cycle(lines)
    history = 100
    fig = plt.figure()
    
    t=np.arange(0,total_samples,10)
    
    sum_rate_performance_FP = []
    sum_rate_performance_random = []
    sum_rate_performance_max = []
    sum_rate_performance_FPMulti_delayedbyone = []
    sum_rate_performance_policy = []
    sum_rate_performance_wmmse = []
    sum_rate_performance_policy = []
    sum_rate_performance_policy2 = []
    ep_start = 0
    for i in range(len(t)):
        if t[i] % options['train_episodes']['T_train'] == 0:
            ep_start = t[i]
        sum_rate_performance_FP.append(np.mean(mean_sum_rate_FP[max(ep_start,t[i]-history):t[i]]))
        sum_rate_performance_random.append(np.mean(mean_sum_rate_randomCS_idealFP[max(ep_start,t[i]-history):t[i]]))
        sum_rate_performance_max.append(np.mean(mean_sum_rate_randomCS_randomP[max(ep_start,t[i]-history):t[i]]))
        sum_rate_performance_FPMulti_delayedbyone.append(np.mean(mean_sum_rate_FPMulti_delayedbyone[max(ep_start,t[i]-history):t[i]]))
        sum_rate_performance_policy.append(np.mean(mean_sum_rate_policy_train_innersims[max(ep_start,t[i]-history):t[i]]))
        sum_rate_performance_policy2.append(np.mean(mean_sum_rate_policy_train_innersims2[max(ep_start,t[i]-history):t[i]]))
        
        
    #plt.figure(figsize=(5,5))
    t=np.arange(0,total_samples,10)
    plt.plot(t, np.array(sum_rate_performance_policy)/avg_result_over, label='proposed',linestyle=next(linecycler))# with Multiplier '+str(power_multiplier_allsims[i]),linestyle=next(linecycler))
    plt.plot(t, np.array(sum_rate_performance_policy2)/avg_result_over, label='joint learning',linestyle=next(linecycler))# with Multiplier '+str(power_multiplier_allsims[i]),linestyle=next(linecycler))
    plt.plot(t, np.array(sum_rate_performance_FP)/avg_result_over, label='ideal FP',linestyle=next(linecycler))
    plt.plot(t, np.array(sum_rate_performance_FPMulti_delayedbyone)/avg_result_over, label='delayed FP',linestyle=next(linecycler))
    plt.plot(t, np.array(sum_rate_performance_random)/avg_result_over, label='random',linestyle=next(linecycler))
    # plt.plot(t, np.array(sum_rate_performance_max)/avg_result_over,'c', label='random CS random P',linestyle=next(linecycler))
    # plt.plot(t, np.array(sum_rate_performance_random)/avg_result_over, linestyle=next(linecycler))
    
    plt.xlabel('training iterations')
    if not pfs:
        plt.ylabel('average spectral efficiency (bps/Hz) per link')
    else:
        plt.ylabel('sum log average rate (ln(bps))')
    plt.grid(True)
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig('./fig/spectraleff_%s_network_%d'%(json_file,overal_sims)+'.pdf', format='pdf', dpi=1000)
    plt.savefig('./fig/spectraleff_%s_network_%d'%(json_file,overal_sims)+'.png', format='png', dpi=1000)
    plt.show(block=False)
    
    # Average performance of the last 200 training slots.
    history = 200
    print('Deployment: %s; policy: %s; K: %d; N: %d'%(json_file,json_file_policy,N,K))
    print('Averages for last %d episodes:'%(history))
    
    if not pfs:
        res_label = 'Sum rate per link'
    else:
        res_label = 'Sum log average rate'
        
    print('%s - proposed: %.2f'%(res_label,np.mean(mean_sum_rate_policy_train_innersims[total_samples-history:])/avg_result_over))
    print('%s - joint learning: %.2f'%(res_label,np.mean(mean_sum_rate_policy_train_innersims2[total_samples-history:])/avg_result_over))
    print('%s - FP: %.2f'%(res_label,np.mean(mean_sum_rate_FP[total_samples-history:])/avg_result_over))
    print('%s - FP Multi delayed: %.2f'%(res_label,np.mean(mean_sum_rate_FPMulti_delayedbyone[total_samples-history:])/avg_result_over))
    print('%s - random: %.2f'%(res_label,np.mean(mean_sum_rate_randomCS_idealFP[total_samples-history:])/avg_result_over))
    print('%s - full: %.2f'%(res_label,np.mean(mean_sum_rate_randomCS_randomP[total_samples-history:])/avg_result_over))
    
    # Average time statistics
    # print('Average time for an FP run: %.2f ms'%(1000 * np.mean(mean_time_FP)))
    # print('Average time for a policy agent to determine its action %.2f ms'%(1000 * np.mean(mean_time_calculating_strategy_takes)))
    # print('Average time for a policy mini-batch train %.2f ms'%(1000 * np.mean(mean_time_optimization_at_each_slot_takes)))
    # print('2 Average time for a policy agent to determine its action %.2f ms'%(1000 * np.mean(mean_time_calculating_strategy_takes2)))
    # print('2 Average time for a policy mini-batch train %.2f ms'%(1000 * np.mean(mean_time_optimization_at_each_slot_takes2)))
    print('Average FP iterations per run: %.2f'%(np.mean(mean_iterations_FP)))


    
if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='give test scenarios.')

    parser.add_argument('--json-file', type=str, default='train_K5_N20_M1_shadow10_episode4-5000_travelIND_fd10',
                        help='json file for the deployment')
    

    parser.add_argument('--json-file-policy', type=str, default='ddpg200_100_50',
                        help='json file for the hyperparameters')
    parser.add_argument('--json-file-CS', type=str, default='dqn100_50_50',
                        help='json file for the hyperparameters')
    parser.add_argument('--json-file-policy2', type=str, default='dqn200_200_100',
                        help='json file for the hyperparameters')
    parser.add_argument('--json-file-CS2', type=str, default='dqn200_200_100',
                        help='json file for the hyperparameters')
    parser.add_argument('--num-sim', type=int, default=0,
                       help='If set to -1, it uses num_simulations of the json file. If set to positive, it runs one simulation with the given id.')
    
    args = parser.parse_args()
    
    test_scenario = {'json_file':args.json_file,
                     'json_file_policy':args.json_file_policy,
                     'json_file_CS':args.json_file_CS,
                     'json_file_policy2':args.json_file_policy2,
                     'json_file_CS2':args.json_file_CS2,
                     'num_sim':args.num_sim}
    scenario = test_scenario
    main(scenario)
    
    
