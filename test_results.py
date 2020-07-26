# -*- coding: utf-8 -*-
"""
@author: anonymous
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import json
import copy
import matplotlib
# matplotlib.use('Qt5Agg')
import argparse
        
def main(args):
    scenario = {}
    json_file_policy_train = args.json_file_PA_train
    json_file_policy_CS_train = args.json_file_CS_train
    
    json_file = args.json_file
    json_file_train = args.json_file_train
    
    scenario['plot'] = args.plot
    
    with open ('./config/deployment/'+json_file+'.json','r') as f:
        options = json.load(f)
    with open ('./config/deployment/'+json_file_train+'.json','r') as f:
        options_train = json.load(f)
    tot_train_episodes = int(options_train['simulation']['total_samples']/options_train['train_episodes']['T_train'])        
    
    ## Kumber of samples
    total_samples = options['simulation']['total_samples']
    
    
    N = options['simulation']['N']
    
    
    
    # simulation parameters
    train_episodes = options['train_episodes']
    mobility_params = options['mobility_params']
    mobility_params['alpha_angle'] = options['mobility_params']['alpha_angle_rad'] * np.pi #radian/sec
    history = 250
    
    
    total_train_samples = train_episodes['T_train']
    mean_p_FP = np.zeros(total_train_samples)
    mean_time_FP = np.zeros(total_train_samples)
    mean_iterations_FP = np.zeros(total_train_samples)
    mean_sum_rate_FP = np.zeros(total_train_samples)
    
    
    mean_sum_rate_FPMulti_delayedbyone = np.zeros(total_train_samples)
    mean_sum_rate_simrandomCS_idealFP = np.zeros(total_train_samples)
    mean_sum_rate_simrandomCS_randomP = np.zeros(total_train_samples)
    
    all_sum_rate_FP = np.zeros(total_samples)
    all_sum_rate_FPMulti_delayedbyone = np.zeros(total_samples)
    all_sum_rate_simrandomCS_idealFP = np.zeros(total_samples)
    all_sum_rate_simrandomCS_randomP = np.zeros(total_samples)
        
    
    # Get the benchmarks.
    file_path = './simulations/sumrate/benchmarks/%s_network%d'%(json_file,0)
    data = np.load(file_path+'.npz')
    p_FP            = data['arr_0']
    alpha_FP        = data['arr_1']
    time_stats_FP   = data['arr_2']
    sum_rate_FP     = data['arr_3']
    
    sum_rate_FPMulti_delayedbyone          = data['arr_4']
    sum_rate_simrandomCS_idealFP         = data['arr_5']
    sum_rate_simrandomCS_randomP            = data['arr_5']
    
    
    
    #Average over each testing episode
    tot_test_episodes = int(total_samples/train_episodes['T_train'])
    for test_ep in range(tot_test_episodes):
        cursor1 = int(test_ep*train_episodes['T_train'])
        cursor2 = int((test_ep+1)*train_episodes['T_train'])
        mean_p_FP = mean_p_FP + np.sum(p_FP[cursor1:cursor2,:],1)/float(tot_test_episodes)
        mean_time_FP = mean_time_FP + time_stats_FP[cursor1:cursor2,0]/float(tot_test_episodes)
        mean_iterations_FP = mean_iterations_FP + time_stats_FP[cursor1:cursor2,1]/float(tot_test_episodes)
        mean_sum_rate_FP = mean_sum_rate_FP + sum_rate_FP[cursor1:cursor2]/float(tot_test_episodes)
    
        
        mean_sum_rate_FPMulti_delayedbyone = mean_sum_rate_FPMulti_delayedbyone + sum_rate_FPMulti_delayedbyone[cursor1:cursor2]/float(tot_test_episodes)
        mean_sum_rate_simrandomCS_idealFP = mean_sum_rate_simrandomCS_idealFP + sum_rate_simrandomCS_idealFP[cursor1:cursor2]/float(tot_test_episodes)
        mean_sum_rate_simrandomCS_randomP = mean_sum_rate_simrandomCS_randomP + sum_rate_simrandomCS_randomP[cursor1:cursor2]/float(tot_test_episodes)
        
        all_sum_rate_FP = copy.copy(sum_rate_FP)
        all_sum_rate_FPMulti_delayedbyone = copy.copy(sum_rate_FPMulti_delayedbyone)
        all_sum_rate_simrandomCS_idealFP = copy.copy(sum_rate_simrandomCS_idealFP)
        all_sum_rate_simrandomCS_randomP = copy.copy(sum_rate_simrandomCS_randomP)
    
    file_path = './simulations/sumrate/test/%s_%s_%s_%s_episode%d.ckpt'%(json_file,json_file_train,json_file_policy_train,json_file_policy_CS_train,tot_train_episodes)
    data = np.load(file_path+'.npz')
    # Get test simulation characteristic.
    included_train_episodes = data['arr_6']
    inner_train_networks = [[0]]#data['arr_7']
    
    mean_sum_rate_policy_train_innersims = np.zeros((len(included_train_episodes),total_train_samples))
    mean_p_strategy_all_train_innersims = np.zeros((len(included_train_episodes),total_train_samples))
    all_sum_rate_policy_lastepisode = np.zeros(total_samples)#np.zeros(int(len(inner_train_networks[0])*total_samples))
    
    mean_time_optimization_at_each_slot_takes = []
    mean_time_calculating_strategy_takes = []
    
    for ep in range(len(included_train_episodes)):
        if included_train_episodes[ep] != 0: # Reload data
            file_path = './simulations/sumrate/test/%s_%s_%s_%s_episode%d.ckpt'%(json_file,json_file_train,json_file_policy_train,json_file_policy_CS_train,included_train_episodes[ep])
            data = np.load(file_path+'.npz')        
        # Get the test episode policy results
        sum_rate_distributed_policy_episode     = data['arr_2']
        p_strategy_all_episode                  = data['arr_3']
        time_optimization_at_each_slot_takes    = data['arr_4']
        time_calculating_strategy_takes         = data['arr_5']
        if included_train_episodes[ep] == included_train_episodes[-1]:
            all_sum_rate_policy_lastepisode = copy.copy(sum_rate_distributed_policy_episode.flatten())
        for i_train in range(len(inner_train_networks[0])):
            for test_ep in range(tot_test_episodes):
                cursor1 = int(test_ep*train_episodes['T_train'])
                cursor2 = int((test_ep+1)*train_episodes['T_train'])
                # Average    
                mean_sum_rate_policy_train_innersims[ep,:] = mean_sum_rate_policy_train_innersims[ep,:] + sum_rate_distributed_policy_episode[i_train,cursor1:cursor2]/float(tot_test_episodes*len(inner_train_networks[0]))
                mean_p_strategy_all_train_innersims[ep,:] = mean_p_strategy_all_train_innersims[ep,:] + np.sum(p_strategy_all_episode[i_train,cursor1:cursor2,:],1)/float(tot_test_episodes*len(inner_train_networks[0]))
                mean_time_optimization_at_each_slot_takes.append(time_optimization_at_each_slot_takes)
                mean_time_calculating_strategy_takes.append(time_calculating_strategy_takes)
    scenario['mean_p_FP'] = mean_p_FP
    scenario['mean_time_FP'] = mean_time_FP
    scenario['mean_iterations_FP'] = mean_iterations_FP
    scenario['mean_sum_rate_FP'] = mean_sum_rate_FP
    
    scenario['mean_sum_rate_FPMulti_delayedbyone'] = mean_sum_rate_FPMulti_delayedbyone
    scenario['mean_sum_rate_simrandomCS_idealFP'] = mean_sum_rate_simrandomCS_idealFP
    scenario['mean_sum_rate_simrandomCS_randomP'] = mean_sum_rate_simrandomCS_randomP
    scenario['mean_sum_rate_policy_train_innersims'] = mean_sum_rate_policy_train_innersims
    scenario['mean_p_strategy_all_train_innersims'] = mean_p_strategy_all_train_innersims  
    scenario['mean_time_optimization_at_each_slot_takes'] = mean_time_optimization_at_each_slot_takes
    scenario['mean_time_calculating_strategy_takes'] = mean_time_calculating_strategy_takes
    
    scenario['all_sum_rate_FP'] = all_sum_rate_FP
    scenario['all_sum_rate_FPMulti_delayedbyone'] = all_sum_rate_FPMulti_delayedbyone
    scenario['all_sum_rate_simrandomCS_idealFP'] = all_sum_rate_simrandomCS_idealFP
    scenario['all_sum_rate_simrandomCS_randomP'] = all_sum_rate_simrandomCS_randomP
    scenario['all_sum_rate_policy_lastepisode'] = all_sum_rate_policy_lastepisode
    
    
    if scenario['plot']:
        with open ('./config/deployment/'+json_file+'.json','r') as f:
            options = json.load(f)
        train_episodes = options['train_episodes']
        file_path = './simulations/sumrate/test/%s_%s_%s_%s_episode%d.ckpt'%(json_file,json_file_train,json_file_policy_train,json_file_policy_CS_train,tot_train_episodes)
        data = np.load(file_path+'.npz')
        # Get test simulation characteristic.
        included_train_episodes = data['arr_6']
        inner_train_networks = data['arr_7']
        
        lines = ["-","--",':','-.',':','-.']
        linecycler = cycle(lines)
        fig = plt.figure()
        
        
    scenario['mean_performance_FP'] = np.array([np.mean(scenario['mean_sum_rate_FP'])/float(N)] * len(included_train_episodes))
    scenario['mean_performance_simrandomCS_idealFP'] = np.array([np.mean(scenario['mean_sum_rate_simrandomCS_idealFP'])/float(N)] * len(included_train_episodes))
    scenario['mean_performance_simrandomCS_randomP'] = np.array([np.mean(scenario['mean_sum_rate_simrandomCS_randomP'])/float(N)] * len(included_train_episodes))
    scenario['mean_performance_FPMulti_delayedbyone'] = np.array([np.mean(scenario['mean_sum_rate_FPMulti_delayedbyone'])/float(N)] * len(included_train_episodes))
    
    scenario['mean_performance_policy'] = []
    for i in range(len(included_train_episodes)):
        scenario['mean_performance_policy'].append(np.mean(scenario['mean_sum_rate_policy_train_innersims'][i,:])/float(N))
    scenario['mean_performance_policy'] = np.array(scenario['mean_performance_policy'])
    # performance with respect to FP:
    scenario['percent_performance_FP'] = scenario['mean_performance_FP']/scenario['mean_performance_FP'][0]
    scenario['percent_performance_simrandomCS_idealFP'] = scenario['mean_performance_simrandomCS_idealFP']/scenario['mean_performance_FP'][0]
    scenario['percent_performance_simrandomCS_randomP'] = scenario['mean_performance_simrandomCS_randomP']/scenario['mean_performance_FP'][0]
    scenario['percent_performance_FPMulti_delayedbyone'] = scenario['mean_performance_FPMulti_delayedbyone']/scenario['mean_performance_FP'][0]
    scenario['percent_performance_policy'] = scenario['mean_performance_policy']/scenario['mean_performance_FP'][0]
    if scenario['plot']:
        t=included_train_episodes
        plt.plot(t, np.array(scenario['percent_performance_FP']), label='ideal FP',linestyle=next(linecycler))
        plt.plot(t, np.array(scenario['percent_performance_FPMulti_delayedbyone']), label='central',linestyle=next(linecycler))
        plt.plot(t, np.array(scenario['percent_performance_simrandomCS_idealFP']), label='random',linestyle=next(linecycler))
        plt.plot(t, np.array(scenario['percent_performance_simrandomCS_randomP']),'c', label='full-power',linestyle=next(linecycler))
        plt.plot(t, np.array(scenario['percent_performance_policy']), label='policy',linestyle=next(linecycler))
        
        plt.xlabel('training episodes')
        plt.ylabel('normalized performance')
        plt.grid(True)
        plt.legend(loc=4)
        plt.tight_layout()
        plt.savefig('./fig/episodeeval_%s_%s_%s'%(json_file,json_file_train,json_file_policy_train)+'.pdf', format='pdf', dpi=1000)
        plt.savefig('./fig/episodeeval_%s_%s_%s'%(json_file,json_file_train,json_file_policy_train)+'.png', format='png', dpi=1000)
        plt.show(block=False)
    
        # CDF comparison with the policy resulted after final training episode. 
        lines = ["-","--",':','-.',':','-.']
        linecycler = cycle(lines)
        fig = plt.figure()
        min_edge = np.min([np.min(scenario['all_sum_rate_policy_lastepisode']),np.min(scenario['all_sum_rate_simrandomCS_idealFP']),np.min(scenario['all_sum_rate_simrandomCS_randomP'])])/float(N)
        max_edge = np.max([np.max(scenario['all_sum_rate_FP']),np.max(scenario['all_sum_rate_FP']),np.max(scenario['all_sum_rate_policy_lastepisode'])])/float(N)
        bins_glob = np.linspace(min_edge,max_edge,200)
        #plt.figure(figsize=(5,5))
        
        counts, bin_edges = np.histogram(scenario['all_sum_rate_FP']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/(float(len(scenario['all_sum_rate_simrandomCS_idealFP']))) # Manual normalization
        plt.plot(bin_edges[1:], cdf, label='ideal FP',linestyle=next(linecycler))
        
        counts, bin_edges = np.histogram(scenario['all_sum_rate_FPMulti_delayedbyone']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/(float(len(scenario['all_sum_rate_simrandomCS_idealFP']))) # Manual normalization
        plt.plot(bin_edges[1:], cdf, label='central',linestyle=next(linecycler))
        
        counts, bin_edges = np.histogram(scenario['all_sum_rate_simrandomCS_idealFP']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/float(len(scenario['all_sum_rate_simrandomCS_idealFP'])) # Manual normalization
        plt.plot(bin_edges[1:], cdf, label='random',linestyle=next(linecycler))
        
        counts, bin_edges = np.histogram(scenario['all_sum_rate_simrandomCS_randomP']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/(float(len(scenario['all_sum_rate_simrandomCS_randomP']))) # Manual normalization
        plt.plot(bin_edges[1:], cdf,'c', label='full-power',linestyle=next(linecycler))
        
        counts, bin_edges = np.histogram(scenario['all_sum_rate_policy_lastepisode']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/(float(len(scenario['all_sum_rate_simrandomCS_idealFP']))) # Manual normalization
        plt.plot(bin_edges[1:], cdf, label='final episode policy',linestyle=next(linecycler))
    
    
        plt.xlabel('average spectral efficiency (bps/Hz) per link')
        plt.ylabel('empirical cumulative probability')
        plt.grid(True)
        plt.legend(loc=4)
        plt.tight_layout()
        plt.savefig('./fig/cdf_%s_%s_%s'%(json_file,json_file_train,json_file_policy_train)+'.pdf', format='pdf', dpi=1000)
        plt.savefig('./fig/cdf_%s_%s_%s'%(json_file,json_file_train,json_file_policy_train)+'.png', format='png', dpi=1000)
        plt.show(block=False)
    
        
    # Average performance metrics.
    print('Test: %s; train %s; policy: %s'%(json_file,json_file_train,json_file_policy_train))
    for (i,ep) in enumerate(included_train_episodes):
        print('Sum rate per link for policy after %d episodes: %.2f, percentage: %.2f'%(ep,scenario['mean_performance_policy'][i],scenario['percent_performance_policy'][i]))
    print('Sum rate per link - FP: %.2f, percentage: %.2f'%(scenario['mean_performance_FP'][0],scenario['percent_performance_FP'][0]))
    print('Sum rate per link - FP w delay: %.2f, percentage: %.2f'%(scenario['mean_performance_FPMulti_delayedbyone'][0],scenario['percent_performance_FPMulti_delayedbyone'][0]))
    print('Sum rate per link - random: %.2f, percentage: %.2f'%(scenario['mean_performance_simrandomCS_idealFP'][0],scenario['percent_performance_simrandomCS_idealFP'][0]))
    print('Sum rate per link - full: %.2f, percentage: %.2f'%(scenario['mean_performance_simrandomCS_randomP'][0],scenario['percent_performance_simrandomCS_randomP'][0]))
    
    # Average time statistics
    # print('Average time for an FP run: %.2f ms'%(1000 * np.mean(scenario['mean_time_FP'])))
    # print('Average time for a policy agent to determine its action %.2f ms'%(1000 * np.mean(scenario['mean_time_calculating_strategy_takes'])))
    print('Average FP iterations per run: %.2f'%(np.mean(scenario['mean_iterations_FP'])))

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='give test scenarios.')
    parser.add_argument('--json-file', type=str, default='test_K10_N50_M2_shadow10_episode10-500_travel0_fd10',
                       help='json file for the deployment the policies are tested on')
    parser.add_argument('--json-file-train', type=str, default='train_K10_N50_M2_shadow10_episode4-5000_travelIND_fd10',
                       help='json file for the deployment that the policy is trained on')
    
    
    parser.add_argument('--json-file-PA-train', type=str, default='ddpg200_100_50',
                        help='json file for the hyperparameters')
    parser.add_argument('--json-file-CS-train', type=str, default='dqn100_50_50',
                        help='json file for the hyperparameters')

    
    parser.add_argument('--plot', type=bool, default=False,
                       help='if set to false will not plot anything, just printout the results.')
    
    args = parser.parse_args()
    
    main(args)
        
    