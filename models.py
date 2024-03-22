"""
File for defining the model, optimization/ integration type

Contains:
    - run_model_steadystate: model without integrating R but using the steady state by finding zeros of the differential equation, use this one if there are no toxins
    - run_model_steadystate_sep: model with steady state calculation for resource and toxins, but using two separate dynamical equations
    ATT: run_model_steadystate_sep can only be used whe at least one toxin is present
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  aux import *
import scipy

# ----------------------------------------------------------------------------------------------------
# model without integrating R but using the steady state by finding zeros of the differential equation

def run_model_steadystate(dR, dN, y_init, N_species, N_nut, cp_matrix, tox_matrix, met_matrix, com_matrix, g, m, w, w_t, l, l_t, tau):
    
    # extract initial vectors, save elements vector to use for reinsertion
    N_0 = y_init[:N_species]
    R_0 = y_init[N_species:N_species+N_nut]
    T_0 = y_init[N_species+N_nut:]
    RT_0 = np.concatenate([R_0,T_0])
    RT_init = RT_0.copy()

    # run the simulation for decided time steps and save results
    N = []
    R = []

    diff = N_0
    N_prev = N_0
 
    while((np.abs(diff) > 0.005).any()):

        # solve steady state for nutrients and toxins
        RT_ss = scipy.optimize.least_squares(dR, RT_0, args=(N_0, com_matrix, met_matrix, RT_init[:N_nut], RT_init[N_nut:], tau, w, w_t, l, l_t),bounds=(0,np.inf)).x
        R_ss = RT_ss[:N_nut]
        T_ss = RT_ss[N_nut:]

        # integrate N one step
        N_out = scipy.integrate.solve_ivp(dN, (0,1), N_0, method='RK45', args=(R_ss, T_ss, cp_matrix, tox_matrix, g, l, w, m,tau))
        N_out.y[N_out.y<2]=0

        # update steady state condition 
        diff = N_prev-N_out.y[:, -1]
        N_prev = N_out.y[:, -1]

        # add results
        N.append(N_out.y[:, -1])
        R.append(RT_ss)

        # reinitialize with results
        RT_0 = RT_ss
        N_0 = N_out.y[:, -1]
        

    N, R = np.array(N),np.array(R)

    # Plot Time Series for Species
    plt.figure(figsize=(8, 5))
    plt.title("Time Series for Species")
    plt.xlabel("Time")
    plt.ylabel("Population")

    colors = plt.cm.tab10.colors  

    for i in range(N.shape[1]):                                       
        plt.plot(N[:, i], label=f'Species {cp_matrix.index.tolist()[i]}', color=colors[i], linewidth=1)

    plt.legend()
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()

    # Plot Time Series for Resources
    plt.figure(figsize=(8, 5))
    plt.title("Time Series for Resources")
    plt.xlabel("Time")
    plt.ylabel("Concentration")

    for i in range(R.shape[1]):                                       
      plt.plot(R[:, i], label=f'Resource {com_matrix.columns.tolist()[i]}', color=colors[i], linewidth=1)

    plt.legend()
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()

    return np.array(N), np.array(R)


#-----------------------------------------------------------------------------------------------------------------------------
# model with separate dynamics for R and T

def run_model_steadystate_sep(dR, dT, dN, y_init, N_species, N_nut, cp_matrix, tox_matrix, met_matrix, tprod_matrix, g, m, w, l, tau):
    
    # extract initial vectors, save elements vector to use for reinsertion
    N_0 = y_init[:N_species]
    R_0 = y_init[N_species:N_species+N_nut]
    T_0 = y_init[N_species+N_nut:]
    R_init = R_0.copy()

    # run the simulation for decided time steps and save results
    N = []
    R = []
    T = []

    diff = N_0
    N_prev = N_0

    while((np.abs(diff) > 0.005).any()):

        # solve steady state for nutrients and toxins
        R_ss = scipy.optimize.least_squares(dR, R_0, args=(N_0, cp_matrix, met_matrix, R_init,w, tau, l), bounds=(0,np.inf),loss='huber').x
        T_ss = scipy.optimize.least_squares(dT, T_0, args=(N_0, tox_matrix, tprod_matrix, tau), bounds=(0,np.inf)).x

        # integrate N one step
        N_out = scipy.integrate.solve_ivp(dN, (0,1), N_0, method='RK45', args=(R_ss, T_ss, cp_matrix, tox_matrix, g, l, w, m, tau))
        N_out.y[N_out.y<2]=0

        # check condition for steady state
        diff = N_prev-N_out.y[:, -1]
        N_prev = N_out.y[:, -1]

        # add results
        N.append(N_out.y[:, -1])
        R.append(R_ss)
        T.append(T_ss)
        
        # reinitialize with results
        R_0 = R_ss
        T_0 = T_ss
        N_0 = N_out.y[:, -1]

    N, R, T = np.array(N),np.array(R),np.array(T)

    # Plot Time Series for Species
    plt.figure(figsize=(8, 5))
    plt.title("Time Series for Species")
    plt.xlabel("Time")
    plt.ylabel("Population")

    colors = plt.cm.tab10.colors  

    for i in range(N.shape[1]):                                       
        plt.plot(N[:, i], label=f'Species {cp_matrix.index.tolist()[i]}', color=colors[i], linewidth=1)

    plt.legend()
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()

    # Plot Time Series for Resources
    plt.figure(figsize=(8, 5))
    plt.title("Time Series for Resources")
    plt.xlabel("Time")
    plt.ylabel("Concentration")

    for i in range(R.shape[1]):                                       
      plt.plot(R[:, i], label=f'Resource {cp_matrix.columns.tolist()[i]}', color=colors[i], linewidth=1)

    plt.legend()
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()

    # Plot Time Series for Toxins
    plt.figure(figsize=(8, 5))
    plt.title("Time Series for toxins")
    plt.xlabel("Time")
    plt.ylabel("Concentration")

    for i in range(T.shape[1]):                                       
      plt.plot(T[:, i], label=f'Toxin {tox_matrix.columns.tolist()[i]}', color=colors[i], linewidth=1)

    plt.legend()
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()

    return N,R,T

