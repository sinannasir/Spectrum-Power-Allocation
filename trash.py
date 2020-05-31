# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""
import numpy as np
#import matplotlib.pyplot as plt
import project_backend as pb
import tensorflow as tf
import collections
import copy

class DQN:
    def __init__(self, options,options_DQN,N,Pmax,noise_var):
        tf.reset_default_graph()
        self.DQNtotal_samples = options['simulation']['total_samples']
        self.DQNtrain_episodes = options['train_episodes']
        R_defined = options['simulation']['R_defined']
        self.DQNR = (2.0/np.sqrt(3))*R_defined
        self.DQNN = N
        self.DQNPmax = Pmax
        self.DQNnoise_var = noise_var
        self.DQNtmp_exp_type_1 = []
        self.DQNtmp_exp_type_2 = []
        self.DQNprev_suminterferences = np.zeros(N)
        for i in range(self.DQNN):
            self.DQNtmp_exp_type_1.append(collections.deque([],4))
            self.DQNtmp_exp_type_2.append(collections.deque([],3))
        
        self.DQNnum_output = self.DQNnum_actions = options_DQN['num_actions'] # Kumber of actions
        self.DQNdiscount_factor = options_DQN['discount_factor']
        
        self.DQNN_neighbors = options_DQN['N_neighbors']
        self.DQNnum_input = 6 + 7 * self.DQNN_neighbors
        learning_rate_0 = options_DQN['learning_rate_0']
        learning_rate_decay = options_DQN['learning_rate_decay']
        learning_rate_min = options_DQN['learning_rate_min']
        self.DQNbatch_size = options_DQN['batch_size']
        memory_per_agent = options_DQN['memory_per_agent']
        # epsilon greedy algorithm
        max_epsilon = options_DQN['max_epsilon']
        epsilon_decay = options_DQN['epsilon_decay']
        min_epsilon = options_DQN['min_epsilon']
        # quasi-static target network update
        self.DQNtarget_update_count = options_DQN['target_update_count']
        self.DQNtime_slot_to_pass_weights = options_DQN['time_slot_to_pass_weights'] # 50 slots needed to pass the weights
        n_hidden_1 = options_DQN['n_hiddens'][0]
        n_hidden_2 = options_DQN['n_hiddens'][1]
        n_hidden_3 = options_DQN['n_hiddens'][2]
        scale_R_inner = options_DQN['scale_R_inner']
        scale_R_interf = options_DQN['scale_R_interf']
        scale_g_dB_R = scale_R_inner*self.DQNR
        rb = 200.0
        if(scale_g_dB_R < rb):
            scale_g_dB = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_R))
        else:
            scale_g_dB = - (128.1 + 37.6* np.log10(scale_g_dB_R/rb) + 37.6* np.log10(0.001*rb)) 
        self.DQNscale_gain = np.power(10.0,scale_g_dB/10.0)
        self.DQNinput_placer = np.log10(self.DQNnoise_var/self.DQNscale_gain)
        scale_g_dB_inter_R = scale_R_interf * self.DQNR
        if(scale_g_dB_R < rb):
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_inter_R))
        else:
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(scale_g_dB_inter_R/rb) + 37.6* np.log10(0.001*rb))
        self.DQNscale_gain_interf = np.power(10.0,scale_g_dB_interf/10.0)
        
        # Experience-replay memory size
        self.DQNmemory_len = memory_per_agent*N
        # learning rate
        self.DQNlearning_rate_all = [learning_rate_0]
        for i in range(1,self.DQNtotal_samples):
            if i % self.DQNtrain_episodes['T_train'] == 0:
                self.DQNlearning_rate_all.append(learning_rate_0)
            else:
                self.DQNlearning_rate_all.append(max(learning_rate_min,learning_rate_decay*self.DQNlearning_rate_all[-1]))
    #            learning_rate_all.append(learning_rate_all[-1])
    
        # epsilon greedy algorithm       
        self.DQNepsilon_all=[max_epsilon]
        for i in range(1,self.DQNtotal_samples):
            if i % self.DQNtrain_episodes['T_train'] == 0:
#                if int(i/self.DQNtrain_episodes['T_train']) == (self.DQNtotal_samples/self.DQNtrain_episodes['T_train']-1):
#                    self.DQNepsilon_all.append(0.0) # Test scenario
#                else:
                self.DQNepsilon_all.append(max_epsilon)
            else:
                self.DQNepsilon_all.append(max(min_epsilon,epsilon_decay*self.DQNepsilon_all[-1]))
        
        # Experience replay memory
        self.DQNmemory = {}
        self.DQNmemory['s'] = collections.deque([],self.DQNmemory_len+self.DQNN)
        self.DQNmemory['s_prime'] = collections.deque([],self.DQNmemory_len+self.DQNN)
        self.DQNmemory['rewards'] = collections.deque([],self.DQNmemory_len+self.DQNN)
        self.DQNmemory['actions'] = collections.deque([],self.DQNmemory_len+self.DQNN)
        
        self.DQNprevious_state = np.zeros((self.DQNN,self.DQNnum_input))
        self.DQNprevious_action = np.ones(self.DQNN) * self.DQNnum_actions
       
        # required for session to know whether dictionary is train or test
        self.DQNis_train = tf.placeholder("bool")   

        self.DQNx_policy = tf.placeholder("float", [None, self.DQNnum_input])
        self.DQNy_policy = tf.placeholder("float", [None, 1])
        with tf.name_scope("weights"):
            self.DQNweights_policy = pb.initial_weights (self.DQNnum_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.DQNnum_output)
        with tf.name_scope("target_weights"): 
            self.DQNweights_target_policy = pb.initial_weights (self.DQNnum_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.DQNnum_output)
        with tf.name_scope("tmp_weights"): 
            self.DQNweights_tmp_policy = pb.initial_weights (self.DQNnum_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.DQNnum_output)
        with tf.name_scope("biases"):
            self.DQNbiases_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.DQNnum_output)
        with tf.name_scope("target_biases"): 
            self.DQNbiases_target_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.DQNnum_output)
        with tf.name_scope("tmp_biases"): 
            self.DQNbiases_tmp_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.DQNnum_output)
        # initialize the neural network for each agent
        self.DQNQNN= pb.neural_net(self.DQNx_policy, self.DQNweights_policy, self.DQNbiases_policy)
        self.DQNQNN_target = pb.neural_net(self.DQNx_policy, self.DQNweights_target_policy,
                                            self.DQNbiases_target_policy)
        self.DQNactions_flatten = tf.placeholder(tf.int32, self.DQNbatch_size)
        self.DQNactions_one_hot = tf.one_hot(self.DQNactions_flatten, self.DQNnum_actions, 1.0, 0.0)
        self.DQNsingle_q = tf.reshape(tf.reduce_sum(tf.multiply(self.DQNQNN, self.DQNactions_one_hot), reduction_indices=1),(self.DQNbatch_size,1))
        # loss function is simply least squares cost
        self.DQNloss = tf.reduce_sum(tf.square(self.DQNy_policy - self.DQNsingle_q))
        self.DQNlearning_rate = (tf.placeholder('float'))
        # RMSprop algorithm used
        self.DQNoptimizer = tf.train.RMSPropOptimizer(self.DQNlearning_rate, decay=0.9,
                                              epsilon=1e-10).minimize(self.DQNloss)

        self.DQNinit = tf.global_variables_initializer()
        # quasi-static target update simulation counter = 0
        self.DQNsaver = tf.train.Saver()
        
    def initialize_DQNupdates(self,sess): # Keed to rund this before calling quasi static.
        self.DQNsaver = tf.train.Saver(tf.global_variables())
        self.DQNupdate_class1 = []
        for (w,tmp_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_weights')):
            self.DQNupdate_class1.append(tf.assign(tmp_w,w))
            sess.run(self.DQNupdate_class1[-1])
        for (b,tmp_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_biases')):
            self.DQNupdate_class1.append(tf.assign(tmp_b,b))
            sess.run(self.DQNupdate_class1[-1])
        self.DQNupdate_class2 = []
        for (tmp_w,t_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_weights')):
            self.DQNupdate_class2.append(tf.assign(t_w,tmp_w))
            sess.run(self.DQNupdate_class2[-1])
        for (tmp_b,t_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_biases')):
            self.DQNupdate_class2.append(tf.assign(t_b,tmp_b))
            sess.run(self.DQNupdate_class2[-1])
        self.DQNsimulation_target_update_counter = self.DQNtarget_update_count
        self.DQNprocess_weight_update = False
        self.DQNsimulation_target_pass_counter = self.DQNtime_slot_to_pass_weights
        print('first update')
    
    def check_memory_restart(self,sess,sim):   
        if(sim %self.DQNtrain_episodes['T_train'] == 0 and sim != 0): # Restart experience replay.
            self.DQNmemory = {}
            self.DQNmemory['s'] = collections.deque([],self.DQNmemory_len+self.DQNN)
            self.DQNmemory['s_prime'] = collections.deque([],self.DQNmemory_len+self.DQNN)
            self.DQNmemory['rewards'] = collections.deque([],self.DQNmemory_len+self.DQNN)
            self.DQNmemory['actions'] = collections.deque([],self.DQNmemory_len+self.DQNN)
            
            self.DQNprevious_state = np.zeros((self.DQNN,self.DQNnum_input))
            self.DQNprevious_action = np.ones(self.DQNN) * self.DQNnum_actions
    
    def quasi_static_alg(self,sess,sim):
        # Quasi-static target Algorithm
        # First check whether target network has to be changed.
        self.DQNsimulation_target_update_counter -= 1
        if (self.DQNsimulation_target_update_counter == 0):
            for update_instance in self.DQNupdate_class1:
                sess.run(update_instance)
            self.DQNsimulation_target_update_counter = self.DQNtarget_update_count
            self.DQNprocess_weight_update = True

        if self.DQNprocess_weight_update:
            self.DQNsimulation_target_pass_counter -= 1
        
        if (self.DQNsimulation_target_pass_counter <= 0):
            for update_instance in self.DQNupdate_class2:
                sess.run(update_instance)
            self.DQNprocess_weight_update = False
            self.DQNsimulation_target_pass_counter = self.DQNtime_slot_to_pass_weights
            
    def act(self,sess,current_local_state,sim):
        # Current QNN outputs for all available actions
        current_QNN_outputs = sess.run(self.DQNQNN_target, feed_dict={self.DQNx_policy: current_local_state.reshape(1,self.DQNnum_input), self.DQNis_train: False})
        # epsilon greedy algorithm
        if np.random.rand() < self.DQNepsilon_all[sim]:
            strategy = np.random.randint(self.DQNnum_actions)
        else:
            strategy = np.argmax(current_QNN_outputs)
        return strategy
    
    def act_noepsilon(self,sess,current_local_state,sim):
        # Current QNN outputs for all available actions
        current_QNN_outputs = sess.run(self.DQNQNN_target, feed_dict={self.DQNx_policy: current_local_state.reshape(1,self.DQNnum_input), self.DQNis_train: False})
        return np.argmax(current_QNN_outputs)
    
    def remember(self,agent,current_local_state,current_reward):
        self.DQNmemory['s'].append(copy.copy(self.DQNprevious_state[agent,:]).reshape(self.DQNnum_input))
        self.DQNmemory['s_prime'].append(copy.copy(current_local_state))
        self.DQNmemory['actions'].append(copy.copy(self.DQNprevious_action[agent]))
        self.DQNmemory['rewards'].append(copy.copy(current_reward))
    
    def train(self,sess,sim):
        if len(self.DQNmemory['s']) >= self.DQNbatch_size+self.DQNN:
            # Minus N ensures that experience samples from previous timeslots been used
            idx = np.random.randint(len(self.DQNmemory['rewards'])-self.DQNN,size=self.DQNbatch_size)
            c_QNN_outputs = sess.run(self.DQNQNN_target, feed_dict={self.DQNx_policy: np.array(self.DQNmemory['s_prime'])[idx, :].reshape(self.DQNbatch_size,self.DQNnum_input),
                                                                 self.DQNis_train: False})
            opt_y = np.array(self.DQNmemory['rewards'])[idx] + self.DQNdiscount_factor * np.max(c_QNN_outputs,axis=1)
            actions = np.array(self.DQNmemory['actions'])[idx]
           
            (tmp,tmp_mse) = sess.run([self.DQNoptimizer, self.DQNloss], feed_dict={self.DQNlearning_rate:self.DQNlearning_rate_all[sim],self.DQNactions_flatten:actions,
                                self.DQNx_policy: np.array(self.DQNmemory['s'])[idx, :],
                                self.DQNy_policy: opt_y.reshape(self.DQNbatch_size,1), self.DQNis_train: True})
    
    def equalize(self,sess):
        for update_instance in self.DQNupdate_class1:
            sess.run(update_instance)
        for update_instance in self.DQNupdate_class2:
            sess.run(update_instance)
            
    def save(self,sess,model_destination):
        self.DQNsaver = tf.train.Saver(tf.global_variables())
        save_path = self.DQNsaver.save(sess, model_destination)
        print("Model saved in path: %s" % save_path)
        
    def load(self,sess,model_destination):
        self.DQNsaver = tf.train.Saver(tf.global_variables())
        self.DQNsaver.restore(sess, model_destination)
        print('Model loaded from: %s' %(model_destination))
        
    def local_state(self,sim,agent,p_strategy_all,H_all_2,neighbors,neighbors_in,sum_rate_list_distributed_policy,sims_pos_p):
        current_experiences = np.zeros(self.DQNnum_input)
        if(p_strategy_all[-1][agent]==0):
            current_experiences[0] = 0.0
        else:
            current_experiences[0] = (p_strategy_all[-1][agent])/self.DQNPmax
        current_experiences[1] = np.log10(H_all_2[sim][agent,:][agent]/self.DQNscale_gain)
        
        current_experiences[2] = np.log10(H_all_2[sim-1][agent,:][agent]/self.DQNscale_gain)
        current_experiences[3] = 0.5 * sum_rate_list_distributed_policy[-1].diagonal()[agent]
        if(len(np.where(np.delete(p_strategy_all[-2],agent)==0)[0])!=self.DQNN-1):
            current_experiences[4] = np.log10((self.DQNnoise_var+np.matmul(np.delete(H_all_2[sim-2][agent,:],agent),
                                           np.delete(p_strategy_all[-2],agent)))/(self.DQNscale_gain))
        else:
            current_experiences[4] = self.DQNinput_placer
        if(len(np.where(np.delete(p_strategy_all[-1],agent)==0)[0])!=self.DQNN-1):
            current_experiences[5] = np.log10((self.DQNnoise_var+np.matmul(np.delete(H_all_2[sim-1][agent,:],agent),
                                           np.delete(p_strategy_all[-1],agent)))/(self.DQNscale_gain))                                
        else:
            current_experiences[5] = self.DQNinput_placer      
        if(len(self.DQNtmp_exp_type_1[agent]) == 0):
            if(len(neighbors_in[-2][agent]) != 0):
                self.DQNtmp_exp_type_1[agent].append(np.log10(np.multiply(H_all_2[sim-2][agent,neighbors_in[-2][agent]],p_strategy_all[-2][neighbors_in[-2][agent]])/(self.DQNscale_gain_interf)))
                
                tmp_exp_type_1_index = np.argsort(self.DQNtmp_exp_type_1[agent][-1])[::-1]
                self.DQNtmp_exp_type_1[agent][-1] = self.DQNtmp_exp_type_1[agent][-1][tmp_exp_type_1_index]
                self.DQNtmp_exp_type_1[agent].append(0.5 * sum_rate_list_distributed_policy[-2].diagonal()[neighbors_in[-2][agent]][tmp_exp_type_1_index])
            else:
                self.DQNtmp_exp_type_1[agent].append(np.array([]))
                self.DQNtmp_exp_type_1[agent].append(np.array([]))
            # Append negative numbers if needed
            if (len(self.DQNtmp_exp_type_1[agent][-2]) < self.DQNN_neighbors):
                self.DQNtmp_exp_type_1[agent][-2] = np.append(self.DQNtmp_exp_type_1[agent][-2],(self.DQNN_neighbors - len(self.DQNtmp_exp_type_1[agent][-2]))*[self.DQNinput_placer])
                self.DQNtmp_exp_type_1[agent][-1] = np.append(self.DQNtmp_exp_type_1[agent][-1],(self.DQNN_neighbors - len(self.DQNtmp_exp_type_1[agent][-1]))*[self.DQNinput_placer])
        if(len(neighbors_in[-1][agent]) != 0):
            self.DQNtmp_exp_type_1[agent].append(np.log10(np.multiply(H_all_2[sim-1][agent,neighbors_in[-1][agent]],p_strategy_all[-1][neighbors_in[-1][agent]])/(self.DQNscale_gain_interf)))
            tmp_exp_type_1_index = np.argsort(self.DQNtmp_exp_type_1[agent][-1])[::-1]
            self.DQNtmp_exp_type_1[agent][-1] = self.DQNtmp_exp_type_1[agent][-1][tmp_exp_type_1_index]
            self.DQNtmp_exp_type_1[agent].append(0.5 * sum_rate_list_distributed_policy[-1].diagonal()[neighbors_in[-1][agent]][tmp_exp_type_1_index])
        else:
            self.DQNtmp_exp_type_1[agent].append(np.array([]))
            self.DQNtmp_exp_type_1[agent].append(np.array([]))                   
        # Append negative numbers if needed
        if (len(self.DQNtmp_exp_type_1[agent][-2]) < self.DQNN_neighbors):
            self.DQNtmp_exp_type_1[agent][-2] = np.append(self.DQNtmp_exp_type_1[agent][-2],(self.DQNN_neighbors - len(self.DQNtmp_exp_type_1[agent][-2]))*[self.DQNinput_placer])
            self.DQNtmp_exp_type_1[agent][-1] = np.append(self.DQNtmp_exp_type_1[agent][-1],(self.DQNN_neighbors - len(self.DQNtmp_exp_type_1[agent][-1]))*[-1])
        current_experiences[(6 + 0 * self.DQNN_neighbors):(6 + 1 * self.DQNN_neighbors)] = self.DQNtmp_exp_type_1[agent][-1][:self.DQNN_neighbors]
        current_experiences[(6 + 1 * self.DQNN_neighbors):(6 + 2 * self.DQNN_neighbors)] = self.DQNtmp_exp_type_1[agent][-2][:self.DQNN_neighbors]
        current_experiences[(6 + 2 * self.DQNN_neighbors):(6 + 3 * self.DQNN_neighbors)] = self.DQNtmp_exp_type_1[agent][-3][:self.DQNN_neighbors]
        current_experiences[(6 + 3 * self.DQNN_neighbors):(6 + 4 * self.DQNN_neighbors)] = self.DQNtmp_exp_type_1[agent][-4][:self.DQNN_neighbors]
        
        current_experiences[(6 + 4 * self.DQNN_neighbors):(6 + 5 * self.DQNN_neighbors)] = current_experiences[(6 + 4 * self.DQNN_neighbors):(6 + 5 * self.DQNN_neighbors)] + self.DQNinput_placer
        current_experiences[(6 + 5 * self.DQNN_neighbors):(6 + 6 * self.DQNN_neighbors)] = current_experiences[(6 + 5 * self.DQNN_neighbors):(6 + 6 * self.DQNN_neighbors)] + self.DQNinput_placer
        current_experiences[(6 + 6 * self.DQNN_neighbors):(6 + 7 * self.DQNN_neighbors)] = current_experiences[(6 + 6 * self.DQNN_neighbors):(6 + 7 * self.DQNN_neighbors)] + self.DQNinput_placer
        if(len(neighbors[-1][agent])>0 and p_strategy_all[-1][agent] != 0):
            self.DQNtmp_exp_type_2[agent].append(np.log10(H_all_2[sim-1][np.array(neighbors[-1][agent]),agent]/self.DQNprev_suminterferences[neighbors[-1][agent]]))
            tmp_exp_type_2_index = np.argsort(self.DQNtmp_exp_type_2[agent][-1])[::-1]
            self.DQNtmp_exp_type_2[agent][-1] = self.DQNtmp_exp_type_2[agent][-1][tmp_exp_type_2_index]
                                    
    
            self.DQNtmp_exp_type_2[agent].append(np.log10((H_all_2[sim-1].diagonal()[np.array(neighbors[-1][agent])])/self.DQNscale_gain))
            self.DQNtmp_exp_type_2[agent][-1] = self.DQNtmp_exp_type_2[agent][-1][tmp_exp_type_2_index]
            self.DQNtmp_exp_type_2[agent].append(0.5 * sum_rate_list_distributed_policy[-1].diagonal()[neighbors[-1][agent]][tmp_exp_type_2_index])
            
            if (len(self.DQNtmp_exp_type_2[agent][-2]) < self.DQNN_neighbors):
                self.DQNtmp_exp_type_2[agent][-1] = np.append(self.DQNtmp_exp_type_2[agent][-1],(self.DQNN_neighbors - len(self.DQNtmp_exp_type_2[agent][-1]))*[self.DQNinput_placer])
                self.DQNtmp_exp_type_2[agent][-2] = np.append(self.DQNtmp_exp_type_2[agent][-2],(self.DQNN_neighbors - len(self.DQNtmp_exp_type_2[agent][-2]))*[self.DQNinput_placer])
                self.DQNtmp_exp_type_2[agent][-3] = np.append(self.DQNtmp_exp_type_2[agent][-3],(self.DQNN_neighbors - len(self.DQNtmp_exp_type_2[agent][-3]))*[self.DQNinput_placer])
            current_experiences[(6 + 4 * self.DQNN_neighbors):(6 + 5 * self.DQNN_neighbors)] = self.DQNtmp_exp_type_2[agent][-3][:self.DQNN_neighbors]
            current_experiences[(6 + 5 * self.DQNN_neighbors):(6 + 6 * self.DQNN_neighbors)] = self.DQNtmp_exp_type_2[agent][-2][:self.DQNN_neighbors]
            current_experiences[(6 + 6 * self.DQNN_neighbors):(6 + 7 * self.DQNN_neighbors)] = self.DQNtmp_exp_type_2[agent][-1][:self.DQNN_neighbors]
        elif(sims_pos_p[agent]>0):
            sim_pos_p = sims_pos_p[agent]
            self.DQNtmp_exp_type_2[agent].append(np.log10(H_all_2[sim_pos_p-1][np.array(neighbors[-1][agent]),agent]/self.DQNprev_suminterferences[neighbors[-1][agent]]))
            tmp_exp_type_2_index = np.argsort(self.DQNtmp_exp_type_2[agent][-1])[::-1]
            self.DQNtmp_exp_type_2[agent][-1] = self.DQNtmp_exp_type_2[agent][-1][tmp_exp_type_2_index]
            self.DQNtmp_exp_type_2[agent].append(np.log10((H_all_2[sim-1].diagonal()[np.array(neighbors[-1][agent])])/self.DQNscale_gain))
            self.DQNtmp_exp_type_2[agent][-1] = self.DQNtmp_exp_type_2[agent][-1][tmp_exp_type_2_index]
            self.DQNtmp_exp_type_2[agent].append(0.5 * sum_rate_list_distributed_policy[-1].diagonal()[neighbors[-1][agent]][tmp_exp_type_2_index])
            if (len(self.DQNtmp_exp_type_2[agent][-2]) < self.DQNN_neighbors):
                self.DQNtmp_exp_type_2[agent][-1] = np.append(self.DQNtmp_exp_type_2[agent][-1],(self.DQNN_neighbors - len(self.DQNtmp_exp_type_2[agent][-1]))*[self.DQNinput_placer])
                self.DQNtmp_exp_type_2[agent][-2] = np.append(self.DQNtmp_exp_type_2[agent][-2],(self.DQNN_neighbors - len(self.DQNtmp_exp_type_2[agent][-2]))*[self.DQNinput_placer])
                self.DQNtmp_exp_type_2[agent][-3] = np.append(self.DQNtmp_exp_type_2[agent][-3],(self.DQNN_neighbors - len(self.DQNtmp_exp_type_2[agent][-3]))*[self.DQNinput_placer])                      
            current_experiences[(6 + 4 * self.DQNN_neighbors):(6 + 5 * self.DQNN_neighbors)] = self.DQNtmp_exp_type_2[agent][-3][:self.DQNN_neighbors]
            current_experiences[(6 + 5 * self.DQNN_neighbors):(6 + 6 * self.DQNN_neighbors)] = self.DQNtmp_exp_type_2[agent][-2][:self.DQNN_neighbors]
            current_experiences[(6 + 6 * self.DQNN_neighbors):(6 + 7 * self.DQNN_neighbors)] = self.DQNtmp_exp_type_2[agent][-1][:self.DQNN_neighbors]  
        return current_experiences