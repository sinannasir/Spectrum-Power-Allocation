# -*- coding: utf-8 -*-
"""
@author: anonymous
"""

import numpy as np
import project_backend as pb
from scipy import special
import json
import copy
import argparse


def main(args):

    json_file = args.json_file
    
    
    num_sim = args.num_sim
    with open ('./config/deployment/'+json_file+'.json','r') as f:
        options = json.load(f)
    
    ## Number of samples
    total_samples = options['simulation']['total_samples']
    
    
    K = options['simulation']['K']
    N = options['simulation']['N']
    # Multi channel scenario, M denotes number of channels.
    if'M' in options['simulation']:
        M = options['simulation']['M']
    else: M = 1
    
    equal_number_for_BS = options['simulation']['equal_number_for_BS']
    R_defined = options['simulation']['R_defined']
    R = (2.0/np.sqrt(3))*R_defined
    min_dist = options['simulation']['min_dist']
    dcor = options['simulation']['dcor']
    shadowing_dev = options['simulation']['shadowing_dev']
    # Kow assume each time slot is 1ms and 
    T = options['simulation']['T']
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
    v_c = options['mobility_params']['v_c'] # speed of light
    f_c = options['mobility_params']['f_c']
    mobility_params = options['mobility_params']
    mobility_params['alpha_angle'] = options['mobility_params']['alpha_angle_rad'] * np.pi #radian/sec
    max_doppler = options['mobility_params']['max_doppler'] # Used only when v_max = 0
    #Some defaults
    R = (2.0/np.sqrt(3))*R_defined
    
    for overal_sims in range(simulation,simulation+num_simulations):
        if isTrain:
            np.random.seed(overal_sims)
        else:
            np.random.seed(1000 + overal_sims + N)
        ### gains
        if isTrain:
            gains,TX_loc,RX_loc,TX_xhex, TX_yhex, TX_neighbors,mirrors = pb.get_gains_hexagon_neighbors_shadowinginc (N,K,R,min_dist,total_samples,shadowing_dev,dcor,
                                                                                                          equal_number_for_BS=equal_number_for_BS,draw=False,
                                                                                                          T=T,
                                                                                                          train_episodes = train_episodes,
                                                                                                          mobility_params = mobility_params)
        else:
            gains = [np.zeros((N,N,total_samples))]
            RX_loc = np.zeros((2,N,total_samples))
            mirrors = {}
            mirrors['cell_mapping'] = np.zeros((N,total_samples)).astype(int)
            mirrors['RX_displacement'] = np.zeros((4,N,total_samples))
            mirrors['RX_loc_all'] = np.zeros((2,N,total_samples))
            mirrors['cell_mapping_all'] = np.zeros((N,total_samples)).astype(int)
            mirrors['RX_displacement_all'] = np.zeros((4,N,total_samples))
            tot_test_episodes = int(total_samples/train_episodes['T_train'])
            for ep in range(tot_test_episodes):
                i_gains,TX_loc,i_RX_loc,TX_xhex, TX_yhex, TX_neighbors,i_mirrors = pb.get_gains_hexagon_neighbors_shadowinginc (N,K,R,min_dist,train_episodes['T_train'],shadowing_dev,dcor,
                                                                                                              equal_number_for_BS=equal_number_for_BS,draw=False,
                                                                                                              T=T,
                                                                                                              train_episodes = train_episodes,
                                                                                                              mobility_params = mobility_params)        
                cursor1 = int(ep*train_episodes['T_train'])
                cursor2 = int((ep+1)*train_episodes['T_train'])
                gains[0][:,:,cursor1:cursor2] = copy.copy(i_gains[0])
                RX_loc[:,:,cursor1:cursor2] = copy.copy(i_RX_loc)
                mirrors['cell_mapping'][:,cursor1:cursor2] = copy.copy(i_mirrors['cell_mapping'])
                mirrors['RX_displacement'][:,:,cursor1:cursor2] = copy.copy(i_mirrors['RX_displacement'])
                mirrors['RX_loc_all'][:,:,cursor1:cursor2] = copy.copy(i_mirrors['RX_loc_all'])
                mirrors['cell_mapping_all'][:,cursor1:cursor2] = copy.copy(i_mirrors['cell_mapping_all'])
                mirrors['RX_displacement_all'][:,:,cursor1:cursor2] = copy.copy(i_mirrors['RX_displacement_all'])
                
        rayleigh_var = 1.0
        
        weights = []
        for loop in range(total_samples):
            weights.append(np.array(np.ones(N)))
       
        ## Coefficients for shadowing
        if max_doppler == 'mixed':
            f_d = np.random.uniform(2,15,(total_samples,K,N))
        elif max_doppler == 'independent':
            f_d = total_samples*[1e10]
        elif max_doppler is None:
            f_d = total_samples*[0] # placeholder...
        else:
            f_d = total_samples*[max_doppler]
        
        print('Gains are ready sim %d'%(overal_sims))
        # Init Optimizer results        
        H_all=[]
    #    if overal_sims>-1:
        # Optimum solution with no CSI delay
        # Also extract cell mapping and displacement
        f_d[0]=np.zeros((K,N))
        cell_mapping = mirrors['cell_mapping']
        RX_displacement = mirrors['RX_displacement']
        tmp_channel_b = [pb.get_random_rayleigh_variable(rayleigh_var, N, M, K)] # will return a matrix of K times N.
        tmp_channel = [pb.get_random_rayleigh_variable(rayleigh_var, N, M)] # Just to initialze this array.
        for k in range(N):
            tmp_channel[0][k,:,:] = tmp_channel_b[0][cell_mapping[:,0],k,:]
        tmp_H_all = np.zeros((N,N,M))
        for m in range(M):
            tmp_H_all[:,:,m] = np.multiply(np.sqrt(gains[0][:,:,0]),((abs(tmp_channel[-1][:,:,m]))))
    
        
        H_all.append(tmp_H_all)
        
        for i in range(1,total_samples):      
            if mobility_params['v_max'] == 0:
                if max_doppler == 'independent':
                    correlation = 0.0
                else:
                    correlation = special.j0(2.0*np.pi*f_d[i]*T)
            else:
                f_d[i] = np.zeros((K,N))
                for k in range(N):
                    f_d[i][:,k] = np.sqrt(RX_displacement[0,k,i]**2+RX_displacement[1,k,i]**2)*f_c/(T*v_c)
                correlation = special.j0(2.0*np.pi*f_d[i]*T)
                correlation = np.dstack(tuple([correlation]*M))
            if train_episodes is not None and i % train_episodes['T_train'] == 0:
                tmp_tmp_channel_b = pb.get_random_rayleigh_variable(rayleigh_var, N, M, K)
            else:
                tmp_tmp_channel_b = pb.get_markov_rayleigh_variable(tmp_channel_b[-1],correlation,rayleigh_var, N, M, K)
            tmp_tmp_channel = np.zeros(np.shape(tmp_channel[0]))+0j
            for k in range(N):
                tmp_tmp_channel[k,:] = tmp_tmp_channel_b[cell_mapping[:,i],k]
            tmp_channel_b.append(tmp_tmp_channel_b)
            tmp_H_all = np.zeros((N,N,M))
            for m in range(M):
                tmp_H_all[:,:,m] = np.multiply(np.sqrt(gains[0][:,:,i]),((abs(tmp_tmp_channel[:,:,m]))))
            H_all.append(tmp_H_all)
            tmp_channel.append(tmp_tmp_channel)
        
        # No need to save the deployment. 
        # np_save_path = './simulations/deployment/%s_network%d'%(json_file,overal_sims)
        # if type(mirrors) is dict:
        #     np.savez(np_save_path,options,f_d,gains,TX_loc,RX_loc,TX_xhex,TX_yhex,TX_neighbors,H_all,mirrors['cell_mapping_all'],
        #              mirrors['RX_loc_all'],mirrors['RX_displacement'],mirrors['RX_displacement_all'],mirrors['cell_mapping'])
        # else:
        #     np.savez(np_save_path,options,gains,TX_loc,RX_loc,TX_xhex,TX_yhex,TX_neighbors,mirrors)
        # print('Saved to %s'%(np_save_path))
        
        np_save_path = './simulations/channel/%s_network%d'%(json_file,overal_sims)
        np.savez(np_save_path,options,H_all,f_d)
        print('Saved to %s'%(np_save_path))
        
if __name__ == "__main__": 

    
    parser = argparse.ArgumentParser(description='give test scenarios.')

    parser.add_argument('--json-file', type=str, default='test_K10_N50_M2_shadow10_episode10-500_travel0_fd10',
                        help='json file for the deployment')
    parser.add_argument('--num-sim', type=int, default=0,
                       help='If set to -1, it uses num_simulations of the json file. If set to positive, it runs one simulation with the given id.')
    
    args = parser.parse_args()
    main(args)


