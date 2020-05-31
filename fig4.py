# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import json
import copy
import matplotlib
matplotlib.use('Qt5Agg')
import argparse

def main(test_scenarios):
    for scenario in test_scenarios:
        json_file = scenario['json_file']
        json_file_train = scenario['json_file_train']
        json_file_train2 = scenario['json_file_train2']
        json_file_policy_train = scenario['json_file_policy_train']
        
        with open ('./config/deployment/'+json_file+'.json','r') as f:
            options = json.load(f)
        
        ## Kumber of samples
        total_samples = options['simulation']['total_samples']
        
        
        N = options['simulation']['N']
        
        # simulation parameters
        train_episodes = options['train_episodes']
        mobility_params = options['mobility_params']
        mobility_params['alpha_angle'] = options['mobility_params']['alpha_angle_rad'] * np.pi #radian/sec
        
        
        
        total_train_samples = train_episodes['T_train']
        mean_p_FP = np.zeros(total_train_samples)
        mean_time_FP = np.zeros(total_train_samples)
        mean_iterations_FP = np.zeros(total_train_samples)
        mean_sum_rate_FP = np.zeros(total_train_samples)
        mean_p_WMMSE = np.zeros(total_train_samples)
        mean_time_WMMSE = np.zeros(total_train_samples)
        mean_iterations_WMMSE = np.zeros(total_train_samples)
        mean_sum_rate_WMMSE = np.zeros(total_train_samples)
        
        mean_sum_rate_delayed_central = np.zeros(total_train_samples)
        mean_sum_rate_random = np.zeros(total_train_samples)
        mean_sum_rate_max = np.zeros(total_train_samples)
        
        all_sum_rate_FP = np.zeros(total_samples)
        all_sum_rate_delayed_central = np.zeros(total_samples)
        all_sum_rate_random = np.zeros(total_samples)
        all_sum_rate_max = np.zeros(total_samples)
        all_sum_rate_WMMSE = np.zeros(total_samples)
            
        
        # Get the benchmarks.
        file_path = './simulations/sumrate/benchmarks/%s_network%d'%(json_file,0)
        data = np.load(file_path+'.npz')
        p_FP            = data['arr_0']
        time_stats_FP   = data['arr_1']
        sum_rate_FP     = data['arr_2']
        p_WMMSE         = data['arr_3']
        time_stats_WMMSE= data['arr_4']
        sum_rate_WMMSE  = data['arr_5'] # 5
        sum_rate_delayed_central          = data['arr_6']
        sum_rate_random         = data['arr_7']
        sum_rate_max            = data['arr_8']
        
        #Average over each testing episode
        tot_test_episodes = int(total_samples/train_episodes['T_train'])
        for test_ep in range(tot_test_episodes):
            cursor1 = int(test_ep*train_episodes['T_train'])
            cursor2 = int((test_ep+1)*train_episodes['T_train'])
            mean_p_FP = mean_p_FP + np.sum(p_FP[cursor1:cursor2,:],1)/float(tot_test_episodes)
            mean_time_FP = mean_time_FP + time_stats_FP[cursor1:cursor2,0]/float(tot_test_episodes)
            mean_iterations_FP = mean_iterations_FP + time_stats_FP[cursor1:cursor2,1]/float(tot_test_episodes)
            mean_sum_rate_FP = mean_sum_rate_FP + sum_rate_FP[cursor1:cursor2]/float(tot_test_episodes)
            mean_p_WMMSE = mean_p_WMMSE + np.sum(p_WMMSE[cursor1:cursor2,:],1)/float(tot_test_episodes)
            mean_time_WMMSE = mean_time_WMMSE + time_stats_WMMSE[cursor1:cursor2,0]/float(tot_test_episodes)
            mean_iterations_WMMSE = mean_iterations_WMMSE + time_stats_WMMSE[cursor1:cursor2,1]/float(tot_test_episodes)
            mean_sum_rate_WMMSE = mean_sum_rate_WMMSE + sum_rate_WMMSE[cursor1:cursor2]/float(tot_test_episodes)
            
            mean_sum_rate_delayed_central = mean_sum_rate_delayed_central + sum_rate_delayed_central[cursor1:cursor2]/float(tot_test_episodes)
            mean_sum_rate_random = mean_sum_rate_random + sum_rate_random[cursor1:cursor2]/float(tot_test_episodes)
            mean_sum_rate_max = mean_sum_rate_max + sum_rate_max[cursor1:cursor2]/float(tot_test_episodes)
            
            all_sum_rate_FP = copy.copy(sum_rate_FP)
            all_sum_rate_delayed_central = copy.copy(sum_rate_delayed_central)
            all_sum_rate_random = copy.copy(sum_rate_random)
            all_sum_rate_max = copy.copy(sum_rate_max)
            all_sum_rate_WMMSE = copy.copy(sum_rate_WMMSE)
        
        file_path = './simulations/sumrate/test/%s_%s_%s_episode%d.ckpt'%(json_file,json_file_train,json_file_policy_train,0)
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
                file_path = './simulations/sumrate/test/%s_%s_%s_episode%d.ckpt'%(json_file,json_file_train,json_file_policy_train,included_train_episodes[ep])
                data = np.load(file_path+'.npz')        
            # Get the test episode policy results
            sum_rate_distributed_policy_episode                  = data['arr_2']
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
    
        
        file_path = './simulations/sumrate/test/%s_%s_%s_episode%d.ckpt'%(json_file,json_file_train,json_file_policy_train,0)
        data2 = np.load(file_path+'.npz')
        # Get test simulation characteristic.
        included_train_episodes2 = data2['arr_6']        
        mean_sum_rate_policy_train_innersims2 = np.zeros((len(included_train_episodes2),total_train_samples))
        mean_p_strategy_all_train_innersims2 = np.zeros((len(included_train_episodes2),total_train_samples))
        all_sum_rate_policy_lastepisode2 = np.zeros(total_samples)#np.zeros(int(len(inner_train_networks[0])*total_samples))
        
        mean_time_optimization_at_each_slot_takes2 = []
        mean_time_calculating_strategy_takes2 = []
        
        for ep in range(len(included_train_episodes2)):
            if included_train_episodes2[ep] != 0: # Reload data
                file_path = './simulations/sumrate/test/%s_%s_%s_episode%d.ckpt'%(json_file,json_file_train2,json_file_policy_train,included_train_episodes[ep])
                data2 = np.load(file_path+'.npz')        
            # Get the test episode policy results
            sum_rate_distributed_policy_episode2                  = data2['arr_2']
            p_strategy_all_episode2                  = data2['arr_3']
            time_optimization_at_each_slot_takes2    = data2['arr_4']
            time_calculating_strategy_takes2         = data2['arr_5']
            if included_train_episodes2[ep] == included_train_episodes2[-1]:
                all_sum_rate_policy_lastepisode2 = copy.copy(sum_rate_distributed_policy_episode2.flatten())
            for i_train in range(len(inner_train_networks[0])):
                for test_ep in range(tot_test_episodes):
                    cursor1 = int(test_ep*train_episodes['T_train'])
                    cursor2 = int((test_ep+1)*train_episodes['T_train'])
                    # Average    
                    mean_sum_rate_policy_train_innersims2[ep,:] = mean_sum_rate_policy_train_innersims2[ep,:] + sum_rate_distributed_policy_episode2[i_train,cursor1:cursor2]/float(tot_test_episodes*len(inner_train_networks[0]))
                    mean_p_strategy_all_train_innersims2[ep,:] = mean_p_strategy_all_train_innersims2[ep,:] + np.sum(p_strategy_all_episode2[i_train,cursor1:cursor2,:],1)/float(tot_test_episodes*len(inner_train_networks[0]))
                    mean_time_optimization_at_each_slot_takes2.append(time_optimization_at_each_slot_takes2)
                    mean_time_calculating_strategy_takes2.append(time_calculating_strategy_takes2)
        
        scenario['mean_p_FP'] = mean_p_FP
        scenario['mean_time_FP'] = mean_time_FP
        scenario['mean_iterations_FP'] = mean_iterations_FP
        scenario['mean_sum_rate_FP'] = mean_sum_rate_FP
        scenario['mean_p_WMMSE'] = mean_p_WMMSE
        scenario['mean_time_WMMSE'] = mean_time_WMMSE
        scenario['mean_iterations_WMMSE'] = mean_iterations_WMMSE
        scenario['mean_sum_rate_WMMSE'] = mean_sum_rate_WMMSE
        
        scenario['mean_sum_rate_delayed_central'] = mean_sum_rate_delayed_central
        scenario['mean_sum_rate_random'] = mean_sum_rate_random
        scenario['mean_sum_rate_max'] = mean_sum_rate_max
        scenario['mean_sum_rate_policy_train_innersims'] = mean_sum_rate_policy_train_innersims
        scenario['mean_p_strategy_all_train_innersims'] = mean_p_strategy_all_train_innersims  
        scenario['mean_time_optimization_at_each_slot_takes'] = mean_time_optimization_at_each_slot_takes
        scenario['mean_time_calculating_strategy_takes'] = mean_time_calculating_strategy_takes
        
        scenario['mean_sum_rate_policy_train_innersims2'] = mean_sum_rate_policy_train_innersims2
        scenario['mean_p_strategy_all_train_innersims2'] = mean_p_strategy_all_train_innersims2  
        scenario['mean_time_optimization_at_each_slot_takes2'] = mean_time_optimization_at_each_slot_takes2
        scenario['mean_time_calculating_strategy_takes2'] = mean_time_calculating_strategy_takes2
        
        scenario['all_sum_rate_FP'] = all_sum_rate_FP
        scenario['all_sum_rate_delayed_central'] = all_sum_rate_delayed_central
        scenario['all_sum_rate_random'] = all_sum_rate_random
        scenario['all_sum_rate_max'] = all_sum_rate_max
        scenario['all_sum_rate_WMMSE'] = all_sum_rate_WMMSE
        scenario['all_sum_rate_policy_lastepisode'] = all_sum_rate_policy_lastepisode
        scenario['all_sum_rate_policy_lastepisode2'] = all_sum_rate_policy_lastepisode2
    
    # For each scenario do testing performance vs training episode analysis.
    for scenario in test_scenarios:
        json_file = scenario['json_file']
        json_file_train = scenario['json_file_train']
        json_file_policy_train = scenario['json_file_policy_train']
        with open ('./config/deployment/'+json_file+'.json','r') as f:
            options = json.load(f)
        train_episodes = options['train_episodes']
        
        file_path = './simulations/sumrate/test/%s_%s_%s_episode%d.ckpt'%(json_file,json_file_train,json_file_policy_train,0)
        data = np.load(file_path+'.npz')
        # Get test simulation characteristic.
        included_train_episodes = data['arr_6']
        inner_train_networks = data['arr_7']
        
        lines = ["-","--",':','-.',':','-.']
        linecycler = cycle(lines)
        fig = plt.figure(figsize=(6,3))
        
        
        scenario['mean_performance_FP'] = np.array([np.mean(scenario['mean_sum_rate_FP'])/float(N)] * len(included_train_episodes))
        scenario['mean_performance_random'] = np.array([np.mean(scenario['mean_sum_rate_random'])/float(N)] * len(included_train_episodes))
        scenario['mean_performance_max'] = np.array([np.mean(scenario['mean_sum_rate_max'])/float(N)] * len(included_train_episodes))
        scenario['mean_performance_delayed_central'] = np.array([np.mean(scenario['mean_sum_rate_delayed_central'])/float(N)] * len(included_train_episodes))
        scenario['mean_performance_wmmse'] = np.array([np.mean(scenario['mean_sum_rate_WMMSE'])/float(N)] * len(included_train_episodes))
        
        scenario['mean_performance_policy'] = []
        for i in range(len(included_train_episodes)):
            scenario['mean_performance_policy'].append(np.mean(scenario['mean_sum_rate_policy_train_innersims'][i,:])/float(N))
        scenario['mean_performance_policy'] = np.array(scenario['mean_performance_policy'])
        
        scenario['mean_performance_policy2'] = []
        for i in range(len(included_train_episodes)):
            scenario['mean_performance_policy2'].append(np.mean(scenario['mean_sum_rate_policy_train_innersims2'][i,:])/float(N))
        scenario['mean_performance_policy2'] = np.array(scenario['mean_performance_policy2'])
        # performance with respect to WMMSE:
        scenario['percent_performance_FP'] = scenario['mean_performance_FP']/scenario['mean_performance_wmmse'][0]
        scenario['percent_performance_random'] = scenario['mean_performance_random']/scenario['mean_performance_wmmse'][0]
        scenario['percent_performance_max'] = scenario['mean_performance_max']/scenario['mean_performance_wmmse'][0]
        scenario['percent_performance_delayed_central'] = scenario['mean_performance_delayed_central']/scenario['mean_performance_wmmse'][0]
        scenario['percent_performance_wmmse'] = scenario['mean_performance_wmmse']/scenario['mean_performance_wmmse'][0]
        scenario['percent_performance_policy'] = scenario['mean_performance_policy']/scenario['mean_performance_wmmse'][0]
        scenario['percent_performance_policy2'] = scenario['mean_performance_policy2']/scenario['mean_performance_wmmse'][0]
        
        t=included_train_episodes
        wwidth = 1.5
        plt.plot(t, np.array(scenario['percent_performance_wmmse']), label='WMMSE w perfect CSI',linestyle=next(linecycler))
        plt.plot(t, np.array(scenario['percent_performance_FP']), label='FP w perfect CSI',linestyle=next(linecycler),linewidth=wwidth)
        plt.plot(t, np.array(scenario['percent_performance_policy']), label='pre-trained policy w mobility',linewidth=wwidth, marker='d',markersize=4)
        plt.plot(t, np.array(scenario['percent_performance_policy2']), label='pre-trained policy w/o mobility',linestyle=next(linecycler),linewidth=1.25*wwidth, marker='x',markersize=4)
        plt.plot(t, np.array(scenario['percent_performance_delayed_central']), label='FP w 1-slot delayed CSI',linestyle=next(linecycler),linewidth=wwidth)
        plt.plot(t, np.array(scenario['percent_performance_random']), label='random power',linestyle=next(linecycler),linewidth=wwidth)
        plt.plot(t, np.array(scenario['percent_performance_max']), label='full power',linestyle=next(linecycler),linewidth=wwidth)
    
        
        plt.xlabel('training episodes')
        plt.ylabel('normalized sum-rate performance')
        plt.grid(True)
        plt.xticks(np.arange(11))
        plt.yticks((np.arange(0.35,1.05,0.05)))
        plt.legend(loc=4)
        plt.tight_layout()
        plt.savefig('./fig/episodeeval_%s_%s_%s'%(json_file,json_file_train,json_file_policy_train)+'.pdf', format='pdf', dpi=1000)
        plt.savefig('./fig/episodeeval_%s_%s_%s'%(json_file,json_file_train,json_file_policy_train)+'.png', format='png', dpi=1000)
        plt.show(block=False)
    
        # CDF comparison with the policy resulted after final training episode. 
        lines = ["-","--",':','-.',':','-.']
        linecycler = cycle(lines)
        fig = plt.figure(figsize=(6,3))
        min_edge = np.min([np.min(scenario['all_sum_rate_policy_lastepisode']),np.min(scenario['all_sum_rate_random']),np.min(scenario['all_sum_rate_max'])])/float(N)
        max_edge = np.max([np.max(scenario['all_sum_rate_FP']),np.max(scenario['all_sum_rate_WMMSE']),np.max(scenario['all_sum_rate_policy_lastepisode'])])/float(N)
        bins_glob = np.linspace(min_edge,max_edge,200)
        #plt.figure(figsize=(5,5))
        counts, bin_edges = np.histogram(scenario['all_sum_rate_WMMSE']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/(float(len(scenario['all_sum_rate_random']))) # Manual normalization
        plt.plot(bin_edges[1:], cdf, label='WMMSE',linestyle=next(linecycler))
        
        counts, bin_edges = np.histogram(scenario['all_sum_rate_FP']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/(float(len(scenario['all_sum_rate_random']))) # Manual normalization
        plt.plot(bin_edges[1:], cdf, label='FP',linestyle=next(linecycler))
        
        counts, bin_edges = np.histogram(scenario['all_sum_rate_delayed_central']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/(float(len(scenario['all_sum_rate_random']))) # Manual normalization
        plt.plot(bin_edges[1:], cdf, label='FP w delay',linestyle=next(linecycler))
        
        counts, bin_edges = np.histogram(scenario['all_sum_rate_random']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/float(len(scenario['all_sum_rate_random'])) # Manual normalization
        plt.plot(bin_edges[1:], cdf, label='random',linestyle=next(linecycler))
        
        counts, bin_edges = np.histogram(scenario['all_sum_rate_max']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/(float(len(scenario['all_sum_rate_max']))) # Manual normalization
        plt.plot(bin_edges[1:], cdf,'c', label='full-power',linestyle=next(linecycler))
        
        counts, bin_edges = np.histogram(scenario['all_sum_rate_policy_lastepisode']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/(float(len(scenario['all_sum_rate_random']))) # Manual normalization
        plt.plot(bin_edges[1:], cdf, label='final episode policy mobility',linestyle=next(linecycler))
        
        counts, bin_edges = np.histogram(scenario['all_sum_rate_policy_lastepisode2']/float(N), bins=bins_glob)
        cdf = np.cumsum(counts)/(float(len(scenario['all_sum_rate_random']))) # Manual normalization
        plt.plot(bin_edges[1:], cdf, label='final episode policy steady',linestyle=next(linecycler))
    
    
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
            print('Sum rate policy after %d episodes: %.2f, percentage: %.2f'%(ep,scenario['mean_performance_policy'][i],scenario['percent_performance_policy'][i]))
        print('Test: %s; train %s; policy: %s'%(json_file,json_file_train2,json_file_policy_train))
        for (i,ep) in enumerate(included_train_episodes):
            print('Sum rate policy after %d episodes: %.2f, percentage: %.2f'%(ep,scenario['mean_performance_policy2'][i],scenario['percent_performance_policy2'][i]))
    
        print('Sum rate per link - WMMSE: %.2f, percentage: %.2f'%(scenario['mean_performance_wmmse'][0],scenario['percent_performance_wmmse'][0]))
        print('Sum rate per link - FP: %.2f, percentage: %.2f'%(scenario['mean_performance_FP'][0],scenario['percent_performance_FP'][0]))
        print('Sum rate per link - FP w delay: %.2f, percentage: %.2f'%(scenario['mean_performance_delayed_central'][0],scenario['percent_performance_delayed_central'][0]))
        print('Sum rate per link - random: %.2f, percentage: %.2f'%(scenario['mean_performance_random'][0],scenario['percent_performance_random'][0]))
        print('Sum rate per link - max: %.2f, percentage: %.2f'%(scenario['mean_performance_max'][0],scenario['percent_performance_max'][0]))
        
        # Average time statistics
        print('Average time for a WMMSE run: %.2f ms'%(1000 * np.mean(scenario['mean_time_WMMSE'])))
        print('Average time for an FP run: %.2f ms'%(1000 * np.mean(scenario['mean_time_FP'])))
        print('Average time for a DDPGnomobility agent to determine its action %.2f ms'%(1000 * np.mean(scenario['mean_time_calculating_strategy_takes2'])))
        print('Average WMMSE iterations per run: %.2f'%(np.mean(scenario['mean_iterations_WMMSE'])))
        print('Average FP iterations per run: %.2f'%(np.mean(scenario['mean_iterations_FP'])))
        
    print('If in the interactive mode, press ctrl+z to exit.')
        
if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description='give test scenarios.')
    parser.add_argument('--json-file', type=str, default='test_K10_N20_shadow10_episode5-2500_travel0_vmax2_5',
                       help='json file for the deployment the policies are tested on')
    parser.add_argument('--json-file-wmobility', type=str, default='train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5',
                       help='json file for the deployment the first policy is trained on')
    parser.add_argument('--json-file-womobility', type=str, default='train_K10_N20_shadow10_episode10-5000_travel0_fd10',
                       help='json file for the deployment the second policy is trained on')
    parser.add_argument('--json-file-policy-train', type=str, default='ddpg200_100_50',
                       help='json file for the hyperparameters')
    
    args = parser.parse_args()
    
    test_scenario = [{'json_file':args.json_file,
                   'json_file_train':args.json_file_wmobility,
                   'json_file_train2':args.json_file_womobility,
                   'json_file_policy_train':args.json_file_policy_train}]
    main(test_scenario)