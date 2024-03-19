"""
File to specify auxiliary functions to __init__.py

Contains:
    - out_replenishment: function to specify that nutrients are replenished externally with a rate imposed as a parameter
    - up_function: function to specify how uptake of nutrients depends on concentration and base uptake rates
    - J_in   : function to specify how uptake is converted into energy flux
    - J_growth: function to specify the fraction of input fluxes that is retained for growth
    - J_out  : function to specify how much of the input energy fluxes is used for nutrients production and the allocation of energy to different nutrients
    ------------------------------------------------------------------------------------------------------------------------------------
    - dNdt: differential equation for species 
    - dR_ss: function to use when wanting to solve for nutrients steady state

"""
import pandas            as pd
import numpy             as np
import scipy.integrate   as scint
import matplotlib.pyplot as plt

# out_replenishment: function to specify that nutrients are replenished externally with a rate imposed as a parameter
# takes            : tau, vector of reinsertion rates; R_0, vector of initial concentrations; R, state vector of concentrations 
# returns          : vector with reinsertion quantities for each nutrient 

def out_replenishment(tau, R_0, R):                                                          
  return (1/tau)*(R_0-R)

# up_function: function to specify how uptake of nutrients depends on concentration and base uptake rates i.e. the input fluxes of nutrients
# takes      : cp_matrix, matrix of base uptake rates that has species as rows and nutrients as columns; R, state vector of concentrations
# returns    : matrix of the same dimensions as cp_matrix with uptake of each nutrient and species as entries

def up_function(cp_matrix, R, sig_max):                                                  
  return np.maximum(0, cp_matrix * R / (1 + cp_matrix * R / sig_max))

# J_in   : function to specify how uptake is converted into energy flux
# takes  : cp_matrix, matrix of base uptake rates that has species as rows and nutrients as columns; R, state vector of concentrations; w, externally fixed vectors with parameters for the conversion of energy flux to nutrient flux
# returns: matrix of the same dimensions as cp_matrix with energy input flux to each species for each nutrient

def J_in(cp_matrix, R, w, sig_max):                                                         
  return w*up_function(cp_matrix, R, sig_max)  

# J_growth: function to specify the fraction of input fluxes that is retained for growth
# takes   : cp_matrix, matrix of base uptake rates that has species as rows and nutrients as columns; R, state vector of concentrations; l, vector with externally fixed leakage rates for each nutrient
# returns : matrix of the same dimensions as cp_matrix with fraction of energy input that is retained for growth                               

def J_growth(cp_matrix, R, l, w, sig_max):                                                      
  return (1-l)*J_in(cp_matrix ,R, w, sig_max)

# J_out  : function to specify how much of the input energy fluxes is used for nutrients production and the allocation of energy to different nutrients
# takes  : cp_matrix, matrix of base uptake rates that has species as rows and nutrients as columns; R, state vector of concentrations; l, vector with externally fixed leakage rates for each nutrient; met_matrix, metabolic matrix specifying allocation
# returns: matrix of the same dimensions as cp_matrix with the output fluxes of each nutrient from each species

def J_out(cp_matrix, R, l, w, met_matrix, sig_max):                                            
  return np.dot(l*J_in(cp_matrix, R, w, sig_max), met_matrix.T)

#----------------------------------------------------------------------------------------------------------------

# dNdt   : function defining the differential equation describing the dynamics of species
# takes  : t; time; N, state vector of species; R, state vector of nutrients; cp_matrix, uptake rates matrix; g, intrinsic growth rate vector; m, maintainence energy requirement vector for each species
# returns: vector of dimension number of species, containing differential growth rate for each species 

def dNdt(t, N, R, T, cp_matrix, tox_matrix, g, l, w, m, k, tau_dil, sig_max):

  # treat extintion
  N_masked = N.copy()
  N_masked[N < 10] = 0

  # calculate inhibition term
  inhibition = (tox_matrix/k)*T

  return g*N_masked*(np.sum(J_growth(cp_matrix,R,l, w, sig_max),axis=1)-m-np.sum(inhibition,axis=1)) - 1/tau_dil*N_masked


# dR_ss  : differential function to use for steady state implementation of nutrients dynamics
# takes  : R, state vector for nutrients; N, state vector of species; cp_matrix, uptake rates matrix; met_matrix, metabolic matrix; R_0, initial nutrients concentration; tau, reinsertion times; w, energy conversion factor; l, leakage rates
# returns: steady state vector of nutrients

def dR_ss(RT, N, com_matrix, met_matrix, R_0, T_0, tau_r, tau_t, tau_dil, w_r, w_t, l_r, l_t, sig_max):

    # combine R and T
    RT_0 = np.concatenate([R_0,T_0])
    tau = np.concatenate([tau_r,tau_t])
    l = np.concatenate([l_r,l_t])
    w = np.concatenate([w_r,w_t])

    # treat complete depletion
    R_masked = RT.copy()
    R_masked[RT < 1] = 0
    
    # also add dilution
    return out_replenishment(tau, RT_0, R_masked)-np.dot((J_in(com_matrix, R_masked, w, sig_max)/w).T,N)+np.dot((J_out(com_matrix, R_masked, l, w, met_matrix, sig_max)/w).T,N)-(1/tau_dil)*R_masked

# dT_ss_sep: 
# takes    : 
# returns  : 

def dT_ss_sep(T, N, inh_matrix, tprod_matrix, tau_dil, sig_max):

    # treat complete depletion
    T_masked = T.copy()
    T_masked[T < 1] = 0

    # also add dilution
    return -np.dot(up_function(inh_matrix, T_masked, sig_max).T,N)+np.dot(tprod_matrix.T,N)/(1+np.dot(tprod_matrix.T,N))-(1/tau_dil)*T_masked

# dR_ss_sep: 
# takes    : 
# returns  :

def dR_ss_sep(R, N, cp_matrix, met_matrix, R_0, w, tau, tau_dil, l, sig_max):

    # treat complete depletion
    R_masked = R.copy()
    R_masked[R < 1] = 0

    met_matrix_res = met_matrix.iloc[:len(R_masked),:len(R_masked)]
    met_matrix_res = met_matrix_res/np.sum(met_matrix_res, axis=0)
    
    # also add dilution
    return out_replenishment(tau, R_0, R_masked)-np.dot((J_in(cp_matrix, R_masked, w, sig_max)/w).T,N)+np.dot((J_out(cp_matrix, R_masked, l, w, met_matrix_res, sig_max)/w).T,N)-(1/tau_dil)*R_masked

