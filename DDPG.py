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

class DDPG:
    def __init__(self, options,options_DDPG,options_DQN,N,M,Pmax,noise_var, seed=None):
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
        
        self.DDPGnum_output = self.DDPGnum_actions = 1 # Kumber of actions
        self.discount_factor = options_DDPG['discount_factor']
        
        self.N_neighbors = options_DDPG['N_neighbors']
        if self.N_neighbors > self.N - 1:
            self.N_neighbors = self.N - 1
        # self.DDPGnum_input = 6 + 7 * self.N_neighbors
        self.DDPGnum_input = 4 + 7 * self.N_neighbors
        if self.pfs: self.DDPGnum_input += (1+ 2 * self.N_neighbors)
        if self.M > 1: self.DDPGnum_input += (1+ 2 * self.N_neighbors)
        
        learning_rate_0 = options_DDPG['learning_rate_0_critic']
        learning_rate_decay = options_DDPG['learning_rate_decay_critic']
        learning_rate_min = options_DDPG['learning_rate_min_critic']
        self.learning_rate_all_critic = [learning_rate_0]
        for i in range(1,self.total_samples):
            if i % self.train_episodes['T_train'] == 0:
                self.learning_rate_all_critic.append(learning_rate_0)
            else:
                self.learning_rate_all_critic.append(max(learning_rate_min,learning_rate_decay*self.learning_rate_all_critic[-1]))
        learning_rate_0 = options_DDPG['learning_rate_0_actor']
        learning_rate_decay = options_DDPG['learning_rate_decay_actor']
        learning_rate_min = options_DDPG['learning_rate_min_actor']
        self.learning_rate_all_actor = [learning_rate_0]
        for i in range(1,self.total_samples):
            if i % self.train_episodes['T_train'] == 0:
                self.learning_rate_all_actor.append(learning_rate_0)
            else:
                self.learning_rate_all_actor.append(max(learning_rate_min,learning_rate_decay*self.learning_rate_all_actor[-1]))
    
        self.batch_size = options_DDPG['batch_size']
        memory_per_agent = options_DDPG['memory_per_agent']
        # epsilon greedy algorithm
        max_epsilon = options_DDPG['max_epsilon']
        epsilon_decay = options_DDPG['epsilon_decay']
        min_epsilon = options_DDPG['min_epsilon']
        # quasi-static target network update
        self.target_update_count = options_DDPG['target_update_count']
        self.time_slot_to_pass_weights = options_DDPG['time_slot_to_pass_weights'] # 50 slots needed to pass the weights
        n_hidden_1 = options_DDPG['n_hiddens'][0]
        n_hidden_2 = options_DDPG['n_hiddens'][1]
        n_hidden_3 = options_DDPG['n_hiddens'][2]
        scale_R_inner = options_DDPG['scale_R_inner']
        scale_R_interf = options_DDPG['scale_R_interf']
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
        
        self.previous_state = np.zeros((self.N,self.DDPGnum_input))
        self.previous_action = np.ones(self.N) * self.DDPGnum_actions
       
        # required for session to know whether dictionary is train or test
        self.is_train = tf.placeholder("bool")
        
        ##
        # Critic Ketwork
        self.x_s_critic = tf.placeholder("float", [None, self.DDPGnum_input])
        self.x_a_critic = tf.placeholder("float", [None, self.DDPGnum_actions])
        self.y_critic = tf.placeholder("float", [None, 1])
        self.x_s_critic_target = tf.placeholder("float", [None, self.DDPGnum_input])
        self.x_a_critic_target = tf.placeholder("float", [None, self.DDPGnum_actions])
        self.y_critic_target = tf.placeholder("float", [None, 1])
        with tf.name_scope("C_weights"):
            self.weights_critic = pb.initial_weights (self.DDPGnum_input+self.DDPGnum_actions, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.DDPGnum_output, seed=self.seed)
        with tf.name_scope("Ctarget_weights"): 
            self.weights_target_critic = pb.initial_weights (self.DDPGnum_input+self.DDPGnum_actions, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.DDPGnum_output, seed=self.seed)
        with tf.name_scope("C_biases"):
            self.biases_critic = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.DDPGnum_output, seed=self.seed)
        with tf.name_scope("Ctarget_biases"): 
            self.biases_target_critic = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.DDPGnum_output, seed=self.seed)
            
        # initialize the neural network for each agent
        self.critic= pb.critic_net(self.x_s_critic,self.x_a_critic, self.weights_critic, self.biases_critic)
        self.critic_target = pb.critic_net(self.x_s_critic_target,self.x_a_critic_target, self.weights_target_critic,
                                            self.biases_target_critic)
        
        self.action_grads_v = tf.gradients(self.critic, self.x_a_critic)
        self.action_grads = [self.action_grads_v[0]]#/(tf.to_float(tf.shape(self.action_grads_v[0])[0]))]#*self.batch_size)]
        
        # l2_regularizer_loss = 0.001*tf.reduce_sum(tf.pow(self.weights_critic['h2'],2))
        self.critic_loss = tf.nn.l2_loss(self.y_critic_target - self.critic) # + l2_regularizer_loss
        
        self.c_loss = []
        self.c_loss_track = []
        # self.critic_loss = tf.reduce_mean(tf.pow(self.y_critic_target- self.critic,2)) #+ l2_regularizer_loss
        self.critic_learning_rate = (tf.placeholder('float'))
        # self.critic_optimizer = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.critic_loss)
        self.critic_optimizer = tf.train.RMSPropOptimizer(self.critic_learning_rate, decay=0.9,
                                              epsilon=1e-10).minimize(self.critic_loss)
        

        # Actor Ketwork
        self.x_actor = tf.placeholder("float", [None, self.DDPGnum_input])
        self.y_actor = tf.placeholder("float", [None, 1])
        self.x_actor_agent = tf.placeholder("float", [None, self.DDPGnum_input])
        with tf.name_scope("A_weights"):
            self.weights_actor = pb.initial_weights (self.DDPGnum_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.DDPGnum_output, seed=self.seed+100)
        with tf.name_scope("Aagent_weights"): 
            self.weights_target_actor = pb.initial_weights (self.DDPGnum_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.DDPGnum_output, seed=self.seed+100)
        with tf.name_scope("Abroadcast_weights"): 
            self.weights_tmp_actor = pb.initial_weights (self.DDPGnum_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.DDPGnum_output, seed=self.seed+100)
        with tf.name_scope("A_biases"):
            self.biases_actor = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.DDPGnum_output, seed=self.seed+100)
        with tf.name_scope("Aagent_biases"): 
            self.biases_target_actor = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.DDPGnum_output, seed=self.seed+100)
        with tf.name_scope("Abroadcast_biases"): 
            self.biases_tmp_actor = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.DDPGnum_output, seed=self.seed+100)
            
        
            
        # initialize the neural network for each agent
        self.actor= pb.actor_net(self.x_actor, self.weights_actor, self.biases_actor)
        self.actor_agent = pb.actor_net(self.x_actor_agent, self.weights_target_actor,
                                            self.biases_target_actor)
        
        self.critic_gradient = tf.placeholder(tf.float32, [None, self.DDPGnum_output])
        
        self.actor_params = self.get_params('A_')
        self.policy_gradients = tf.gradients(self.actor, self.actor_params, -self.critic_gradient)       
        self.actor_learning_rate = (tf.placeholder('float'))
        # Adam
        # self.actor_optimizer = tf.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(zip(self.policy_gradients,self.actor_params))
        # RMSprop algorithm used
        self.actor_optimizer = tf.train.RMSPropOptimizer(self.actor_learning_rate, decay=0.9,
                                              epsilon=1e-10).apply_gradients(zip(self.policy_gradients,self.actor_params))

        # quasi-static target update simulation counter = 0
        
        # self.std = tf.placeholder("float")
        # self.noise = tf.random_uniform(shape = (1, 1), minval=-self.std, maxval=self.std)
        
        # DQN inititalization
        R_defined = options['simulation']['R_defined']
        
        self.DQNnum_output = self.DQNnum_actions = self.M # Kumber of actions
        self.DQNdiscount_factor = options_DQN['discount_factor']
        
        self.DQNnum_input = self.DDPGnum_input * self.M
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
        scale_g_dB_R = scale_R_inner*self.R
        rb = 200.0
        if(scale_g_dB_R < rb):
            scale_g_dB = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_R))
        else:
            scale_g_dB = - (128.1 + 37.6* np.log10(scale_g_dB_R/rb) + 37.6* np.log10(0.001*rb)) 
        self.DQNscale_gain = np.power(10.0,scale_g_dB/10.0)
        self.DQNinput_placer = np.log10(self.noise_var/self.DQNscale_gain)
        scale_g_dB_inter_R = scale_R_interf * self.R
        if(scale_g_dB_R < rb):
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_inter_R))
        else:
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(scale_g_dB_inter_R/rb) + 37.6* np.log10(0.001*rb))
        self.DQNscale_gain_interf = np.power(10.0,scale_g_dB_interf/10.0)
        
        # Experience-replay memory size
        self.DQNmemory_len = memory_per_agent*N
        # learning rate
        self.DQNlearning_rate_all = [learning_rate_0]
        for i in range(1,self.total_samples):
            if i % self.train_episodes['T_train'] == 0:
                self.DQNlearning_rate_all.append(learning_rate_0)
            else:
                self.DQNlearning_rate_all.append(max(learning_rate_min,learning_rate_decay*self.DQNlearning_rate_all[-1]))
    #            learning_rate_all.append(learning_rate_all[-1])
    
        # epsilon greedy algorithm       
        self.DQNepsilon_all=[max_epsilon]
        for i in range(1,self.total_samples):
            if i % self.train_episodes['T_train'] == 0:
#                if int(i/self.train_episodes['T_train']) == (self.total_samples/self.train_episodes['T_train']-1):
#                    self.DQNepsilon_all.append(0.0) # Test scenario
#                else:
                self.DQNepsilon_all.append(max_epsilon)
            else:
                self.DQNepsilon_all.append(max(min_epsilon,epsilon_decay*self.DQNepsilon_all[-1]))
        
        # Experience replay memory
        self.DQNmemory = {}
        self.DQNmemory['s'] = collections.deque([],self.DQNmemory_len+self.N)
        self.DQNmemory['s_prime'] = collections.deque([],self.DQNmemory_len+self.N)
        self.DQNmemory['rewards'] = collections.deque([],self.DQNmemory_len+self.N)
        self.DQNmemory['actions'] = collections.deque([],self.DQNmemory_len+self.N)
        
        self.DQNprevious_state = np.zeros((self.N,self.DQNnum_input))
        self.DQNprevious_action = np.ones(self.N) * self.DQNnum_actions
       
        # required for session to know whether dictionary is train or test
        self.DQNis_train = tf.placeholder("bool")   

        self.DQNx_policy = tf.placeholder("float", [None, self.DQNnum_input])
        self.DQNy_policy = tf.placeholder("float", [None, 1])
        with tf.name_scope("weights"):
            self.DQNweights_policy = pb.initial_weights (self.DQNnum_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.DQNnum_output, seed=self.seed+200)
        with tf.name_scope("target_weights"): 
            self.DQNweights_target_policy = pb.initial_weights (self.DQNnum_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.DQNnum_output, seed=self.seed+200)
        with tf.name_scope("tmp_weights"): 
            self.DQNweights_tmp_policy = pb.initial_weights (self.DQNnum_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.DQNnum_output, seed=self.seed+200)
        with tf.name_scope("biases"):
            self.DQNbiases_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.DQNnum_output, seed=self.seed+200)
        with tf.name_scope("target_biases"): 
            self.DQNbiases_target_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.DQNnum_output, seed=self.seed+200)
        with tf.name_scope("tmp_biases"): 
            self.DQNbiases_tmp_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.DQNnum_output, seed=self.seed+200)
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

        # quasi-static target update simulation counter = 0
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.global_variables())
        
        # Epsilon-greedy for DDPG:
        self.strategy_translation = np.zeros(10)
        self.strategy_translation[0] = 0.0 # Tx power 0
        Pmax_dB = 38.0-30
        Pmin_dB = 10.0-30
        # Calculate steps in dBm
        self.strategy_translation_dB_step = (Pmax_dB-Pmin_dB)/(10-2)
        for i in range(1,10-1):
            self.strategy_translation[i] = np.power(10.0,((Pmin_dB+(i-1)*self.strategy_translation_dB_step))/10)
        self.strategy_translation[-1] = Pmax
    
    def get_params(self, para_name):
        sets=[]
        for var in tf.trainable_variables():
            if not var.name.find(para_name):
                sets.append(var)
        return sets

    def initialize_critic_updates(self,sess): # Keed to rund this before calling quasi static.
        self.update_class1_critic = []
        for (w,tmp_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='C_weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Ctarget_weights')):
            self.update_class1_critic.append(tf.assign(tmp_w,w))
            sess.run(self.update_class1_critic[-1])
        for (b,tmp_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='C_biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Ctarget_biases')):
            self.update_class1_critic.append(tf.assign(tmp_b,b))
            sess.run(self.update_class1_critic[-1])
        print('first critic update')

    def initialize_actor_updates(self,sess): # Keed to rund this before calling quasi static.
        self.update_class1 = []
        for (w,tmp_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='A_weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Abroadcast_weights')):
            self.update_class1.append(tf.assign(tmp_w,w))
            sess.run(self.update_class1[-1])
        for (b,tmp_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='A_biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Abroadcast_biases')):
            self.update_class1.append(tf.assign(tmp_b,b))
            sess.run(self.update_class1[-1])
        self.update_class2 = []
        for (tmp_w,t_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Abroadcast_weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Aagent_weights')):
            self.update_class2.append(tf.assign(t_w,tmp_w))
            sess.run(self.update_class2[-1])
        for (tmp_b,t_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Abroadcast_biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Aagent_biases')):
            self.update_class2.append(tf.assign(t_b,tmp_b))
            sess.run(self.update_class2[-1])
        self.simulation_target_update_counter = self.target_update_count
        self.process_weight_update = False
        self.simulation_target_pass_counter = self.time_slot_to_pass_weights
        print('first update')
        
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
        if(sim %self.train_episodes['T_train'] == 0 and sim != 0): # Restart experience replay.
            self.memory = {}
            self.memory['s'] = collections.deque([],self.memory_len+self.N)
            self.memory['s_prime'] = collections.deque([],self.memory_len+self.N)
            self.memory['rewards'] = collections.deque([],self.memory_len+self.N)
            self.memory['actions'] = collections.deque([],self.memory_len+self.N)
            
            self.previous_state = np.zeros((self.N,self.DDPGnum_input))
            self.previous_action = np.ones(self.N) * self.DDPGnum_actions
            
            # DQN memory reset
            self.DQNmemory = {}
            self.DQNmemory['s'] = collections.deque([],self.DQNmemory_len+self.N)
            self.DQNmemory['s_prime'] = collections.deque([],self.DQNmemory_len+self.N)
            self.DQNmemory['rewards'] = collections.deque([],self.DQNmemory_len+self.N)
            self.DQNmemory['actions'] = collections.deque([],self.DQNmemory_len+self.N)
            
            self.DQNprevious_state = np.zeros((self.N,self.DQNnum_input))
            self.DQNprevious_action = np.ones(self.N) * self.DQNnum_actions
    
    def update_handler(self,sess,sim):
        # First check whether target network has to be changed.
        self.simulation_target_update_counter -= 1
        # Update critic all the time after training
        for update_instance in self.update_class1_critic:
            sess.run(update_instance)
        # Actor broadcast
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
        
        # DQN
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
            
    def PAact(self,sess,current_local_state,sim,forcezero=False):
        # for stability
        if forcezero and np.random.rand() < self.epsilon_all[0]:
            return np.random.rand()        
        if sim < 500:
            if np.random.rand() < self.epsilon_all[0]:
                strategy = np.random.rand()
                return strategy
        else:
            if np.random.rand() < self.epsilon_all[sim]:
                strategy = np.random.rand()
                return strategy
        strategy = sess.run(self.actor_agent, feed_dict={self.x_actor_agent: current_local_state.reshape(1,self.DDPGnum_input), self.is_train: False})[0][0]

        return strategy#_noise
    
    def PAact_noepsilon(self,sess,current_local_state,sim):
        # Current QNN outputs for all available actions
        return sess.run(self.actor_agent, feed_dict={self.x_actor_agent: current_local_state.reshape(1,self.DDPGnum_input), self.is_train: False})[0][0]
    
    def PAremember(self,agent,current_local_state,current_reward):
        self.memory['s'].append(copy.copy(self.previous_state[agent,:]).reshape(self.DDPGnum_input))
        self.memory['s_prime'].append(copy.copy(current_local_state))
        self.memory['actions'].append(copy.copy(self.previous_action[agent]))
        self.memory['rewards'].append(copy.copy(current_reward))
        
    def CSact(self,sess,current_local_state,sim):
        # Current QNN outputs for all available actions
        current_QNN_outputs = sess.run(self.DQNQNN_target, feed_dict={self.DQNx_policy: current_local_state.reshape(1,self.DQNnum_input), self.DQNis_train: False})
        # epsilon greedy algorithm
        if np.random.rand() < self.DQNepsilon_all[sim] or sim < 500:
            strategy = np.random.randint(self.DQNnum_actions)
        else:
            strategy = np.argmax(current_QNN_outputs)
        return strategy
    
    def CSact_noepsilon(self,sess,current_local_state,sim):
        # Current QNN outputs for all available actions
        current_QNN_outputs = sess.run(self.DQNQNN_target, feed_dict={self.DQNx_policy: current_local_state.reshape(1,self.DQNnum_input), self.DQNis_train: False})
        return np.argmax(current_QNN_outputs)
    
    def CSremember(self,agent,current_local_state,current_reward):
        self.DQNmemory['s'].append(copy.copy(self.DQNprevious_state[agent,:]).reshape(self.DQNnum_input))
        self.DQNmemory['s_prime'].append(copy.copy(current_local_state))
        self.DQNmemory['actions'].append(copy.copy(self.DQNprevious_action[agent]))
        self.DQNmemory['rewards'].append(copy.copy(current_reward))
    
    def train(self,sess,sim):
        # skip training for 100 time slots.
        # if sim < 100: return
        if len(self.memory['s']) >= self.batch_size+self.N:
            # Minus N ensures that experience samples from previous timeslots been used
            idx = np.random.randint(len(self.memory['rewards'])-self.N,size=self.batch_size)
            s_prime_shaped = np.array(self.memory['s_prime'])[idx, :].reshape(self.batch_size,self.DDPGnum_input)
            action_t_1_batch = sess.run(self.actor_agent, feed_dict={self.x_actor_agent: s_prime_shaped})
            #Q'(s_i+1,a_i+1)
            q_t_1 = sess.run(self.critic_target, feed_dict={self.x_s_critic_target: s_prime_shaped,
                                                            self.x_a_critic_target: action_t_1_batch,self.is_train: False})
            if not self.pfs:
                y_batch = np.array(self.memory['rewards'])[idx].reshape(self.batch_size,1) + self.discount_factor * q_t_1
            else:
                y_batch = ((np.array(self.memory['rewards'])[idx]-np.mean(self.memory['rewards']))/np.std(self.memory['rewards'])).reshape(self.batch_size,1) + self.discount_factor * q_t_1
            
            s_shaped = np.array(self.memory['s'])[idx, :].reshape(self.batch_size,self.DDPGnum_input)
            (tmp,tmp_critloss) = sess.run([self.critic_optimizer, self.critic_loss], feed_dict={self.critic_learning_rate:self.learning_rate_all_critic[sim],
                                self.x_s_critic: s_shaped,
                                self.x_a_critic: np.array(self.memory['actions'])[idx].reshape(self.batch_size,self.DDPGnum_actions),
                                self.y_critic_target: y_batch.reshape(self.batch_size,1), self.is_train: True})
            self.c_loss_track.append(tmp_critloss)
            if sim%100==0:
                self.c_loss.append(np.mean(self.c_loss_track))
                self.c_loss_track = []
            # if sim%5==0:
            action_for_delQ = sess.run(self.actor, feed_dict={self.x_actor:s_shaped})    
            del_Q_a = sess.run(self.action_grads, feed_dict={self.x_s_critic: s_shaped,
                                                                 self.x_a_critic: action_for_delQ,self.is_train: False})[0]
            tmp = sess.run([self.actor_optimizer], feed_dict={self.actor_learning_rate:self.learning_rate_all_actor[sim],
                                                              self.x_actor: s_shaped, 
                                                self.critic_gradient: del_Q_a, self.is_train: True})
        # Train DQN
        if self.M > 1 and len(self.DQNmemory['s']) >= self.DQNbatch_size+self.N:
            # Minus N ensures that experience samples from previous timeslots been used
            idx = np.random.randint(len(self.DQNmemory['rewards'])-self.N,size=self.DQNbatch_size)
            c_QNN_outputs = sess.run(self.DQNQNN_target, feed_dict={self.DQNx_policy: np.array(self.DQNmemory['s_prime'])[idx, :].reshape(self.DQNbatch_size,self.DQNnum_input),
                                                                 self.DQNis_train: False})
            if self.pfs:
                opt_y = ((np.array(self.DQNmemory['rewards'])[idx]-np.mean(self.DQNmemory['rewards']))/np.std(self.DQNmemory['rewards'])) + self.DQNdiscount_factor * np.max(c_QNN_outputs,axis=1)
            else:
                opt_y = np.array(self.DQNmemory['rewards'])[idx] + self.DQNdiscount_factor * np.max(c_QNN_outputs,axis=1)

            actions = np.array(self.DQNmemory['actions'])[idx]
           
            (tmp,tmp_mse) = sess.run([self.DQNoptimizer, self.DQNloss], feed_dict={self.DQNlearning_rate:self.DQNlearning_rate_all[sim],self.DQNactions_flatten:actions,
                                self.DQNx_policy: np.array(self.DQNmemory['s'])[idx, :],
                                self.DQNy_policy: opt_y.reshape(self.DQNbatch_size,1), self.DQNis_train: True})
            

    
    def equalize(self,sess):
        for update_instance in self.update_class1:
            sess.run(update_instance)
        for update_instance in self.update_class2:
            sess.run(update_instance)
        for update_instance in self.DQNupdate_class1:
            sess.run(update_instance)
        for update_instance in self.DQNupdate_class2:
            sess.run(update_instance)
            
    def save(self,sess,model_destination):
        self.saver = tf.train.Saver(tf.global_variables())
        save_path = self.saver.save(sess, model_destination)
        print("Model saved in path: %s" % save_path)
        
    def load(self,sess,model_destination):
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(sess, model_destination)
        print('Model loaded from: %s' %(model_destination))
        
    def local_state(self,sim,agent,p_strategy_all,alpha_strategy_all,alpha_int_strategy_all,H_all_2,sum_rate_list_distributed_policy,weights):
        global_state = np.zeros(self.DQNnum_input)
        local_state_input = self.DQNnum_input // self.M
        for m in range(self.M):
            global_state[m*local_state_input:(m+1)*local_state_input] = self.local_state_singlechannel(sim,agent,m,alpha_strategy_all,alpha_int_strategy_all,p_strategy_all,H_all_2,sum_rate_list_distributed_policy,weights)
        return global_state


    def local_state_singlechannel(self,sim,agent,m,alpha_strategy_all,alpha_int_strategy_all,p_strategy_all,H_all_2,sum_rate_list_distributed_policy,weights):
        state = np.zeros(self.DQNnum_input // self.M)
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
    
    

