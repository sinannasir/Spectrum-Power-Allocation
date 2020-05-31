# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""

import json

train1 = {'simulation':{
'num_simulations' : 10,
'simulation_index_start' : 0,
'N' : 10,
'K' : 20,
'R_defined' : 400,
'min_dist' : 35,
'dcor' : 10,
'T' : 20e-3,
'total_samples' : 100000,
'isTrain' : True,
'equal_number_for_BS' : True},

'train_episodes':{
'T_train' : 5000,
'T_sleep' : 5000,
'cell_passing_training' : True,
'cell_passing_sleeping' : True,
'T_register' : 50},

'mobility_params':{
'v_c' : 3e8,
'f_c' : 1e9,
'v_max' : 5.0,
'a_max' : 0.5,
'alpha_angle_rad' : 0.175,
'T_mobility' : 50,
'max_doppler' : None},

'dqn':{
'N_neighbors' : 5,
'scale_R_inner' : 0.75,
'scale_R_interf' : 2.5,
'n_hiddens' : [200,100,40],
'neightresh' : 5,
'discount_factors' : [0.5],
'learning_rate_0' : 0.01,
'learning_rate_decay' : 1-1E-4,
'learning_rate_min' : 0.0,
'target_update_count' : 100,
'time_slot_to_pass_weights' : 50,
'max_epsilon' : 0.25,
'epsilon_decay' : 1-1E-4,
'min_epsilon' : 0.01,
'batch_size' : 256,
'memory_per_agent' : 1000}}


print(json.dumps(train1,indent=4))