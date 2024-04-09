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
    - dT_ss_sep: function to implement resource dynamics separately from toxins
    - dR_ss_sep: function to implement toxins dynamics independently from metabolism

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

def up_function(cp_matrix, R):                                                  
  return np.maximum(0, cp_matrix * R / (1 + R))

# J_in   : function to specify how uptake is converted into energy flux
# takes  : cp_matrix, matrix of base uptake rates that has species as rows and nutrients as columns; R, state vector of concentrations; w, externally fixed vectors with parameters for the conversion of energy flux to nutrient flux
# returns: matrix of the same dimensions as cp_matrix with energy input flux to each species for each nutrient

def J_in(cp_matrix, R, w):                                                         
  return w*up_function(cp_matrix, R)  

# J_growth: function to specify the fraction of input fluxes that is retained for growth
# takes   : cp_matrix, matrix of base uptake rates that has species as rows and nutrients as columns; R, state vector of concentrations; l, vector with externally fixed leakage rates for each nutrient
# returns : matrix of the same dimensions as cp_matrix with fraction of energy input that is retained for growth                               

def J_growth(cp_matrix, R, l, w):                                                      
  return (1-l)*J_in(cp_matrix ,R, w)

# J_out  : function to specify how much of the input energy fluxes is used for nutrients production and the allocation of energy to different nutrients
# takes  : cp_matrix, matrix of base uptake rates that has species as rows and nutrients as columns; R, state vector of concentrations; l, vector with externally fixed leakage rates for each nutrient; met_matrix, metabolic matrix specifying allocation
# returns: matrix of the same dimensions as cp_matrix with the output fluxes of each nutrient from each species

def J_out(cp_matrix, R, l, w, met_matrix):                                            
  return np.dot(l*J_in(cp_matrix, R, w), met_matrix.T)

#----------------------------------------------------------------------------------------------------------------

# dNdt   : function defining the differential equation describing the dynamics of species
# takes  : t; time; N, state vector of species; R, state vector of nutrients; cp_matrix, uptake rates matrix; g, intrinsic growth rate vector; m, maintainence energy requirement vector for each species
# returns: vector of dimension number of species, containing differential growth rate for each species 

def dNdt(t, N, R, T, cp_matrix, tox_matrix, g, l, w, w_t, m, tau):

  # calculate inhibition term
  inhibition = w_t*tox_matrix*T/(1+tox_matrix*T)
  k=20
  
  return g*N*(np.sum(J_growth(cp_matrix,R,l, w)*k,axis=1)-m-np.sum(inhibition*k,axis=1)) - 1/tau*N


# dR_ss  : differential function to use for steady state implementation of nutrients dynamics
# takes  : R, state vector for nutrients; N, state vector of species; cp_matrix, uptake rates matrix; met_matrix, metabolic matrix; R_0, initial nutrients concentration; tau, reinsertion times; w, energy conversion factor; l, leakage rates
# returns: steady state vector of nutrients

def dR_ss(RT, N, com_matrix, met_matrix, R_0, T_0, tau, w_r, w_t, l_r, l_t):

    # combine R and T
    RT_0 = np.concatenate([R_0,T_0]).copy()
    l = np.concatenate([l_r,l_t])
    w = np.concatenate([w_r,w_t])

    # also add dilution
    return out_replenishment(tau, RT_0, RT)-np.dot((J_in(com_matrix, RT, w)/w).T,N)+np.dot((J_out(com_matrix, RT, l, w, met_matrix)/w).T,N)

# dT_ss_sep: function for dynamics of toxins when different from nutrients dynamics, here linear
# takes    : T, state vector of toxins concentrations; N, state vector of species; inh_matrix, inhibition matrix; tprod_matrix, toxins production matrix; tau, dilution time
# returns  : vector with time derivative of toxins concentration

def dT_ss_sep(T, N, inh_matrix, tprod_matrix, tau):

    # also add dilution
    return -np.dot(up_function(inh_matrix, T).T,N)+np.dot(tprod_matrix.T,N)-(1/tau)*T

# dR_ss_sep: function for resource dynamics when different form nutrients dynamics
# takes    : R, state vector of resources; N, state vector of species; cp_matrix, consumer preference matrix; met_matrix, metabolic matrix; R_0, initial concentration of resources; w, vector with energy conversion factors; tau, reinsertion rate; l, vector with leakage rates
# returns  : vector of time derivatives for nutrients

def dR_ss_sep(R, N, cp_matrix, met_matrix, R_0, w, tau, l):

    met_matrix_res = met_matrix.iloc[:len(R),:len(R)]
    met_matrix_res = met_matrix_res/(np.sum(met_matrix_res, axis=0)+1E-6)
    
    # also add dilution
    return out_replenishment(tau, R_0, R)-np.dot((J_in(cp_matrix, R, w)/w).T,N)+np.dot((J_out(cp_matrix, R, l, w, met_matrix_res)/w).T,N)

