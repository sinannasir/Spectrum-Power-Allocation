# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""

import numpy as np
import matplotlib.pyplot as plt
import project_backend as pb
from matplotlib import rcParams
import matplotlib
matplotlib.use('Qt5Agg')

def main(test_scenarios):
    # Number of TXRX
    np.random.seed(0)
    N = 10
    K = 20
    R_defined = 400
    R = (2.0/np.sqrt(3))*R_defined
    min_dist = 35
    
    # Now assume each time slot is 1ms and 
    T = 20e-3
    
    # Number of samples
    total_samples = 15000#int(3600/T)
    
    # simulation parameters
    train_episodes = {'T_train':5000, 'T_sleep':50000, 'cell_passing_training':True, 'cell_passing_sleeping':True, 'T_register':50} # cell passing during sleeping.
    mobility_params = {}
    mobility_params['v_max'] = 2.5 #m/s
    mobility_params['a_max'] = 0.5 #m/s2
    mobility_params['alpha_angle'] = (0.175) * np.pi #radian/sec
    mobility_params['T_mobility'] = 50#*20e-3
    ## Drop moving pairs
    gains,TX_loc,RX_loc,TX_xhex, TX_yhex, TX_neighbors,mirrors  = pb.get_gains_hexagon_neighbors_shadowinginc (K,N,R,min_dist,total_samples,10,10,
                                                                                                              equal_number_for_BS=True,draw=False,
                                                                                                              T=T,
                                                                                                              train_episodes = train_episodes,
                                                                                                              mobility_params = mobility_params)    
    
    # plt.rcParams.update({'font.size': 22})
    RX_loc2 = mirrors['RX_loc_all']
    #plt.figure(figsize=(10,10))
    #t=T*np.arange(history,len(sum_rate_sim_1),10)
    fig = plt.figure()
    # fig = plt.figure(figsize=(15,15))
    ax=fig.add_subplot(1,1,1)
    rcParams.update({'figure.autolayout': True})
    for i in range(1):
        plt.plot(TX_loc[0,i],TX_loc[1,i],'g^', label = 'AP')
        plt.plot(TX_xhex [:,i],TX_yhex [:,i],'k-')
        circ = plt.Circle((TX_loc[0,i],TX_loc[1,i]),min_dist,color='k',fill=False)
        ax.add_patch(circ)
    for i in range(1,N):
        plt.plot(TX_loc[0,i],TX_loc[1,i],'g^')
        plt.plot(TX_xhex [:,i],TX_yhex [:,i],'k-')
        circ = plt.Circle((TX_loc[0,i],TX_loc[1,i]),min_dist,color='k',fill=False)
        ax.add_patch(circ)
    is_train = True
    train_P = train_episodes['T_train']
    travel_P = train_episodes['T_sleep']
    colors = ['r','g','m']
    cursor = 0
    tot_visit_train = 0
    for i in range(K):
        cursor = 0
        tot_visit_train = 0
        while cursor < (np.shape(RX_loc2)[2]):
            if is_train:
                ccursor = cursor // 55000
                if i ==0:
                    plt.plot(RX_loc2[0,i,cursor:cursor+train_P], RX_loc2[1,i,cursor:cursor+train_P],color=colors[ccursor],label='e %d'%(ccursor+1),linewidth=0.5)
                else:
                    plt.plot(RX_loc2[0,i,cursor:cursor+train_P], RX_loc2[1,i,cursor:cursor+train_P],color=colors[ccursor],linewidth=0.5)
                    
                cursor += train_P
                is_train = False
                if tot_visit_train == 7:
                    break
            else:
                if i ==0 and tot_visit_train == 0 :
                    plt.plot(RX_loc2[0,i,cursor:cursor+travel_P], RX_loc2[1,i,cursor:cursor+travel_P],color='b',label='travel',linewidth=0.5)    
                else:
                    plt.plot(RX_loc2[0,i,cursor:cursor+travel_P], RX_loc2[1,i,cursor:cursor+travel_P],color='b',linewidth=0.5)
                cursor += travel_P
                is_train = True
                tot_visit_train+=1
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis('equal')
    plt.tight_layout()
    plt.xlabel('x axis position (meters)')
    plt.ylabel('y axis position (meters)')
    # plt.legend(loc=4)
    plt.savefig('./fig/movementall.pdf', format='pdf', dpi=1000)
    # plt.xlim((min(RX_loc2[0,i,:]),max(RX_loc2[0,i,:])))
    # plt.ylim((min(RX_loc2[1,i,:]),max(RX_loc2[1,i,:])))
    plt.show()
if __name__ == "__main__": main()