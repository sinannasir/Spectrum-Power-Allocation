# -*- coding: utf-8 -*-
"""
@author: anonymous
"""
import numpy as np
#import matplotlib.pyplot as plt
import project_backend as pb
import tensorflow as tf
import collections
import copy

class DQN:
    def __init__(self, options,options_policy,N,M,Pmax,noise_var,seed=None):
        tf.reset_default_graph()
        self.total_samples = options['simulation']['total_samples']
        self.train_episodes = options['train_episodes']
        R_defined = options['simulation']['R_defined']
        self.R = (2.0/np.sqrt(3))*R_defined
        self.N = N
        self.M = M
        self.Pmax = Pmax
        self.noise_var = noise_var
        self.seed = seed

        # PFS set to true means that we save log average sum-rate instead of sum-rate
        self.pfs = False
        if'pfs' in options['simulation']:
            self.pfs = options['simulation']['pfs']
            
            
        # self.tmp_exp_type_1 = []
        # self.tmp_exp_type_2 = []
        self.prev_suminterferences = np.zeros((N,M))
        if self.M > 1:
            self.sorted_channels = np.zeros((N,M))
        # for i in range(self.N):
        #     self.tmp_exp_type_1.append(collections.deque([],4))
        #     self.tmp_exp_type_2.append(collections.deque([],3))
        
        self.num_output = self.num_actions = options_policy['num_actions'] * self.M # Number of actions
        self.power_levels = options_policy['num_actions']
        self.discount_factor = options_policy['discount_factor']
        
        self.N_neighbors = options_policy['N_neighbors']
        if self.N_neighbors > self.N - 1:
            self.N_neighbors = self.N - 1
        self.num_input = (4 + 7 * self.N_neighbors) * self.M
        if self.pfs: self.num_input = (5 + 8 * self.N_neighbors) * self.M
        if self.M > 1: self.num_input += ((1+ 2 * self.N_neighbors))*self.M
        learning_rate_0 = options_policy['learning_rate_0']
        learning_rate_decay = options_policy['learning_rate_decay']
        learning_rate_min = options_policy['learning_rate_min']
        self.batch_size = options_policy['batch_size']
        memory_per_agent = options_policy['memory_per_agent']
        # epsilon greedy algorithm
        max_epsilon = options_policy['max_epsilon']
        epsilon_decay = options_policy['epsilon_decay']
        min_epsilon = options_policy['min_epsilon']
        # quasi-static target network update
        self.target_update_count = options_policy['target_update_count']
        self.time_slot_to_pass_weights = options_policy['time_slot_to_pass_weights'] # 50 slots needed to pass the weights
        n_hidden_1 = options_policy['n_hiddens'][0]
        n_hidden_2 = options_policy['n_hiddens'][1]
        n_hidden_3 = options_policy['n_hiddens'][2]
        scale_R_inner = options_policy['scale_R_inner']
        scale_R_interf = options_policy['scale_R_interf']
        scale_g_dB_R = scale_R_inner*self.R
        rb = 200.0
        if(scale_g_dB_R < rb):
            scale_g_dB = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_R))
        else:
            scale_g_dB = - (128.1 + 37.6* np.log10(scale_g_dB_R/rb) + 37.6* np.log10(0.001*rb)) 
        self.scale_gain = np.power(10.0,scale_g_dB/10.0)
        self.input_placer = np.log10(self.noise_var/self.scale_gain)
        scale_g_dB_inter_R = scale_R_interf * self.R
        if(scale_g_dB_R < rb):
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_inter_R))
        else:
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(scale_g_dB_inter_R/rb) + 37.6* np.log10(0.001*rb))
        self.scale_gain_interf = np.power(10.0,scale_g_dB_interf/10.0)
        
        # Experience-replay memory size
        self.memory_len = memory_per_agent*N
        # learning rate
        self.learning_rate_all = [learning_rate_0]
        for i in range(1,self.total_samples):
            if i % self.train_episodes['T_train'] == 0:
                self.learning_rate_all.append(learning_rate_0)
            else:
                self.learning_rate_all.append(max(learning_rate_min,learning_rate_decay*self.learning_rate_all[-1]))
    #            learning_rate_all.append(learning_rate_all[-1])
    
        # epsilon greedy algorithm       
        self.epsilon_all=[max_epsilon]
        for i in range(1,self.total_samples):
            if i % self.train_episodes['T_train'] == 0:
#                if int(i/self.train_episodes['T_train']) == (self.total_samples/self.train_episodes['T_train']-1):
#                    self.epsilon_all.append(0.0) # Test scenario
#                else:
                self.epsilon_all.append(max_epsilon)
            else:
                self.epsilon_all.append(max(min_epsilon,epsilon_decay*self.epsilon_all[-1]))
        
        # Experience replay memory
        self.memory = {}
        self.memory['s'] = collections.deque([],self.memory_len+self.N)
        self.memory['s_prime'] = collections.deque([],self.memory_len+self.N)
        self.memory['rewards'] = collections.deque([],self.memory_len+self.N)
        self.memory['actions'] = collections.deque([],self.memory_len+self.N)
        
        self.previous_state = np.zeros((self.N,self.num_input))
        self.previous_action = np.ones(self.N) * self.num_actions
       
        # required for session to know whether dictionary is train or test
        self.is_train = tf.placeholder("bool")   

        self.x_policy = tf.placeholder("float", [None, self.num_input])
        self.y_policy = tf.placeholder("float", [None, 1])
        with tf.name_scope("weights"):
            self.weights_policy = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output,seed=self.seed)
        with tf.name_scope("target_weights"): 
            self.weights_target_policy = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output,seed=self.seed)
        with tf.name_scope("tmp_weights"): 
            self.weights_tmp_policy = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output,seed=self.seed)
        with tf.name_scope("biases"):
            self.biases_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output,seed=self.seed)
        with tf.name_scope("target_biases"): 
            self.biases_target_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output,seed=self.seed)
        with tf.name_scope("tmp_biases"): 
            self.biases_tmp_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output,seed=self.seed)
        # initialize the neural network for each agent
        self.QNN= pb.neural_net(self.x_policy, self.weights_policy, self.biases_policy)
        self.QNN_target = pb.neural_net(self.x_policy, self.weights_target_policy,
                                            self.biases_target_policy)
        self.actions_flatten = tf.placeholder(tf.int32, self.batch_size)
        self.actions_one_hot = tf.one_hot(self.actions_flatten, self.num_actions, 1.0, 0.0)
        self.single_q = tf.reshape(tf.reduce_sum(tf.multiply(self.QNN, self.actions_one_hot), reduction_indices=1),(self.batch_size,1))
        # loss function is simply least squares cost
        self.loss = tf.reduce_sum(tf.square(self.y_policy - self.single_q))
        self.learning_rate = (tf.placeholder('float'))
        # RMSprop algorithm used
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9,
                                              epsilon=1e-10).minimize(self.loss)

        self.init = tf.global_variables_initializer()
        # quasi-static target update simulation counter = 0
        self.saver = tf.train.Saver()
        
    def initialize_updates(self,sess): # Keed to rund this before calling quasi static.
        self.saver = tf.train.Saver(tf.global_variables())
        self.update_class1 = []
        for (w,tmp_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_weights')):
            self.update_class1.append(tf.assign(tmp_w,w))
            sess.run(self.update_class1[-1])
        for (b,tmp_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_biases')):
            self.update_class1.append(tf.assign(tmp_b,b))
            sess.run(self.update_class1[-1])
        self.update_class2 = []
        for (tmp_w,t_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_weights')):
            self.update_class2.append(tf.assign(t_w,tmp_w))
            sess.run(self.update_class2[-1])
        for (tmp_b,t_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_biases')):
            self.update_class2.append(tf.assign(t_b,tmp_b))
            sess.run(self.update_class2[-1])
        self.simulation_target_update_counter = self.target_update_count
        self.process_weight_update = False
        self.simulation_target_pass_counter = self.time_slot_to_pass_weights
        print('first update')
    
    def check_memory_restart(self,sess,sim):   
        if(sim %self.train_episodes['T_train'] == 0 and sim != 0): # Restart experience replay.
            self.memory = {}
            self.memory['s'] = collections.deque([],self.memory_len+self.N)
            self.memory['s_prime'] = collections.deque([],self.memory_len+self.N)
            self.memory['rewards'] = collections.deque([],self.memory_len+self.N)
            self.memory['actions'] = collections.deque([],self.memory_len+self.N)
            
            self.previous_state = np.zeros((self.N,self.num_input))
            self.previous_action = np.ones(self.N) * self.num_actions
    
    def quasi_static_alg(self,sess,sim):
        # Quasi-static target Algorithm
        # First check whether target network has to be changed.
        self.simulation_target_update_counter -= 1
        if (self.simulation_target_update_counter == 0):
            for update_instance in self.update_class1:
                sess.run(update_instance)
            self.simulation_target_update_counter = self.target_update_count
            self.process_weight_update = True

        if self.process_weight_update:
            self.simulation_target_pass_counter -= 1
        
        if (self.simulation_target_pass_counter <= 0):
            for update_instance in self.update_class2:
                sess.run(update_instance)
            self.process_weight_update = False
            self.simulation_target_pass_counter = self.time_slot_to_pass_weights
            
    def act(self,sess,current_local_state,sim,forcezero=False):
        # Current QNN outputs for all available actions
        current_QNN_outputs = sess.run(self.QNN_target, feed_dict={self.x_policy: current_local_state.reshape(1,self.num_input), self.is_train: False})
        # epsilon greedy algorithm
        if forcezero and np.random.rand() < self.epsilon_all[0]:
            return np.random.randint(self.num_actions)
        if np.random.rand() < self.epsilon_all[sim]:
            strategy = np.random.randint(self.num_actions)
        else:
            strategy = np.argmax(current_QNN_outputs)
        return strategy
    
    def act_noepsilon(self,sess,current_local_state,sim):
        # Current QNN outputs for all available actions
        current_QNN_outputs = sess.run(self.QNN_target, feed_dict={self.x_policy: current_local_state.reshape(1,self.num_input), self.is_train: False})
        return np.argmax(current_QNN_outputs)
    
    def remember(self,agent,current_local_state,current_reward):
        self.memory['s'].append(copy.copy(self.previous_state[agent,:]).reshape(self.num_input))
        self.memory['s_prime'].append(copy.copy(current_local_state))
        self.memory['actions'].append(copy.copy(self.previous_action[agent]))
        self.memory['rewards'].append(copy.copy(current_reward))
    
    def train(self,sess,sim):
        if len(self.memory['s']) >= self.batch_size+self.N:
            # Minus N ensures that experience samples from previous timeslots been used
            idx = np.random.randint(len(self.memory['rewards'])-self.N,size=self.batch_size)
            c_QNN_outputs = sess.run(self.QNN_target, feed_dict={self.x_policy: np.array(self.memory['s_prime'])[idx, :].reshape(self.batch_size,self.num_input),
                                                                 self.is_train: False})
            opt_y = np.array(self.memory['rewards'])[idx] + self.discount_factor * np.max(c_QNN_outputs,axis=1)
            actions = np.array(self.memory['actions'])[idx]
           
            (tmp,tmp_mse) = sess.run([self.optimizer, self.loss], feed_dict={self.learning_rate:self.learning_rate_all[sim],self.actions_flatten:actions,
                                self.x_policy: np.array(self.memory['s'])[idx, :],
                                self.y_policy: opt_y.reshape(self.batch_size,1), self.is_train: True})
    
    def equalize(self,sess):
        for update_instance in self.update_class1:
            sess.run(update_instance)
        for update_instance in self.update_class2:
            sess.run(update_instance)
            
    def save(self,sess,model_destination):
        self.saver = tf.train.Saver(tf.global_variables())
        save_path = self.saver.save(sess, model_destination)
        print("Model saved in path: %s" % save_path)
        
    def load(self,sess,model_destination):
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(sess, model_destination)
        print('Model loaded from: %s' %(model_destination))

    def local_state(self,sim,agent,p_strategy_all,alpha_strategy_all,H_all_2,sum_rate_list_distributed_policy,weights):
        global_state = np.zeros(self.num_input)
        local_state_input = self.num_input // self.M
        for m in range(self.M):
            global_state[m*local_state_input:(m+1)*local_state_input] = self.local_state_singlechannel(sim,agent,m,alpha_strategy_all,p_strategy_all,H_all_2,sum_rate_list_distributed_policy,weights)
        return global_state
    
    def local_state_singlechannel(self,sim,agent,m,alpha_strategy_all,p_strategy_all,H_all_2,sum_rate_list_distributed_policy,weights):
        state = np.zeros(self.num_input // self.M)
        cursor = 0
        state[cursor] = alpha_strategy_all[-1][agent,m]* p_strategy_all[-1][agent]
        cursor += 1
        if self.pfs:
            state[cursor] = 0.2 * (1.0/ weights[-1][agent] -5.)
            cursor += 1
        state[cursor] = np.log10(H_all_2[sim][agent,agent,m]/self.scale_gain)
        cursor += 1
        if(len(np.where(np.delete(alpha_strategy_all[-1][:,m]*p_strategy_all[-1],agent)==0)[0])!=self.N-1):
            state[cursor] = np.log10((self.noise_var+np.matmul(np.delete(H_all_2[sim][agent,:,m],agent),
                                            np.delete(alpha_strategy_all[-1][:,m]*p_strategy_all[-1],agent)))/(self.scale_gain))                                
        else:
            state[cursor] = self.input_placer
        cursor += 1
            
        state[cursor] = 0.2 * (sum_rate_list_distributed_policy[-1][agent,agent,m] - 5)
        cursor += 1
        
        if self.M > 1:
            state[cursor] = self.sorted_channels[agent,m]
            cursor += 1
        #interferers
        sorted_interferers_all = np.argsort(H_all_2[sim-1][agent,:,m])[::-1]
        sorted_interferers_all = np.delete(sorted_interferers_all,np.where(sorted_interferers_all==agent))
        
        sorted_interferers = np.hstack((np.setdiff1d(sorted_interferers_all,np.where(alpha_strategy_all[-1][:,m]==0),assume_unique=True),
                                        np.setdiff1d(sorted_interferers_all,np.where(alpha_strategy_all[-1][:,m]==1),assume_unique=True)))
        
        
        state[(cursor):(cursor + self.N_neighbors)] = np.log10(H_all_2[sim][agent,sorted_interferers[:self.N_neighbors],m]/self.scale_gain_interf)
        cursor += self.N_neighbors
        state[(cursor):(cursor + self.N_neighbors)] = alpha_strategy_all[-1][sorted_interferers[:self.N_neighbors],m]*p_strategy_all[-1][sorted_interferers[:self.N_neighbors]]
        cursor += self.N_neighbors
        state[(cursor):(cursor + self.N_neighbors)] = 0.2 * (sum_rate_list_distributed_policy[-1][sorted_interferers[:self.N_neighbors],sorted_interferers[:self.N_neighbors],m] - 5)
        cursor += self.N_neighbors
        if self.pfs:
            state[(cursor):(cursor + self.N_neighbors)] = 0.2 * (1.0 / weights[-1][sorted_interferers[:self.N_neighbors]] - 5.)
            cursor += self.N_neighbors
        if self.M > 1:
            state[(cursor):(cursor + self.N_neighbors)] = self.sorted_channels[sorted_interferers[:self.N_neighbors],m]
            cursor += self.N_neighbors
        #interfereds        
        sorted_interfereds_all = np.argsort(H_all_2[sim-1][:,agent,m]/self.prev_suminterferences[:,m])[::-1]
        sorted_interfereds_all = np.delete(sorted_interfereds_all,np.where(sorted_interfereds_all==agent))
        
        sorted_interfereds = np.hstack((np.setdiff1d(sorted_interfereds_all,np.where(alpha_strategy_all[-1][:,m]==0),assume_unique=True),
                                        np.setdiff1d(sorted_interfereds_all,np.where(alpha_strategy_all[-1][:,m]==1),assume_unique=True)))
        
        state[(cursor):(cursor + self.N_neighbors)] = np.log10(H_all_2[sim-1][sorted_interfereds[:self.N_neighbors],sorted_interfereds[:self.N_neighbors],m]/self.scale_gain)
        cursor += self.N_neighbors
        state[(cursor):(cursor + self.N_neighbors)] = np.log10(H_all_2[sim-1][sorted_interfereds[:self.N_neighbors],agent,m]/self.scale_gain_interf)
        cursor += self.N_neighbors
        state[(cursor):(cursor + self.N_neighbors)] = np.log10(self.prev_suminterferences[sorted_interfereds[:self.N_neighbors],m]/self.scale_gain)
        cursor += self.N_neighbors
        state[(cursor):(cursor + self.N_neighbors)] = 0.2 * (sum_rate_list_distributed_policy[-1][sorted_interfereds[:self.N_neighbors],sorted_interfereds[:self.N_neighbors],m] -5)
        cursor += self.N_neighbors
        if self.pfs:
            state[(cursor):(cursor + self.N_neighbors)] = 0.2 * (1.0 / weights[-1][sorted_interfereds[:self.N_neighbors]] - 5.0)
            cursor += self.N_neighbors
        if self.M > 1:
            state[(cursor):(cursor + self.N_neighbors)] = self.sorted_channels[sorted_interfereds[:self.N_neighbors],m]
            cursor += self.N_neighbors
        return state
