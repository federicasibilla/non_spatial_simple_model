"""
code to define different kinds of examples to be used

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from aux import *
from models import *
from make_network import *
import scipy

# ------------------------------------------------------------------------------------------------------
# DEFINE CLASS FOR THE COMMUNITY
class Community:
    def __init__(self,init_state,dynamics,parameters,matrices):

        N_0 = init_state[0]
        R_0 = init_state[1]
        T_0 = init_state[2]

        self.N = N_0.copy()
        self.R = R_0.copy()
        self.T = T_0.copy()

        self.dN = dynamics[0]
        self.dR = dynamics[1]

        self.p = parameters.copy()

        self.cp,self.inh,self.met,self.comb = matrices

    def run_steady_state(self):
        
        return run_model_steadystate(self.dR, self.dN, np.concatenate([self.N,self.R,self.T]), len(self.N), len(self.R),
                                     self.cp,self.inh, self.met, self.comb, self.p['g'], self.p['m'], self.p['w'], self.p['w_t'], self.p['l'], self.p['l_t'],
                                     self.p['k'], self.p['sig_max'], self.p['tau'], self.p['tau_t'], self.p['tau_dil'])

# ------------------------------------------------------------------------------------------------------
# COMPETITION FOR ONE RESOURCE

# nodes dictionary (species)
species_dict_cfr = {'A' : 'specialist1',
                    'B' : 'specialist2',
                    'M' : 'source1',
}
# edges dictionary (nutrients)
# for each nutrients who produces it and all the species that use it
nutrients_dict_cfr = {'x' : ['MA,MB', 0.5, 0.5]                                        
}
# define toxins dictionary
# for each nutrients who produces it and all the species that use it
toxins_dict_cfr = {
}

# make network consumer preferences matrix and metabolic matrix
consumer_preference_matrix_cfr, metabolic_matrix_cfr, inhibition_matrix_cfr, combined_matrix_cfr = make_net_mat(species_dict_cfr, nutrients_dict_cfr, toxins_dict_cfr)

def get_matrices_cfr():
    return [consumer_preference_matrix_cfr, metabolic_matrix_cfr, inhibition_matrix_cfr, combined_matrix_cfr]

# fix parameters
parameters_dict_cfr = { # initial resource availability
                        'R_0_cfr' : np.array([300]),   
                        # initial toxins concentration
                        'T_0_cfr' : np.array([]),
                        # initial species abundance                                                
                        'N_0_cfr' : np.array([100,100,1]), 
                        # inverse reinsertion rate                                                  
                        'tau_cfr' : np.array([1]), 
                        # toxins inverse insetrion rate
                        'tau_t_cfr' : np.array([]),
                        # dilution time
                        'tau_dil_cfr' : 100,
                        # intrinsic growth rates                                              
                        'g_cfr' : np.array([1,0.8,0.1]), 
                        # maintainance energy requirements                                                    
                        'm_cfr' : np.array([0.1,0.1,0.]),  
                        # external leakage of nutrients                                                         
                        'l_cfr' : np.array([0.8]),  
                        # toxins leakage
                        'l_t_cfr' : np.array([]),
                        # energy content of resources                                               
                        'w_cfr' : np.array([1]), 
                        # energy content (negative) in toxins i.e. toxicity (?
                        'w_t_cfr' : np.array([]),
                        # maximal absorption for Monod function
                        'sig_max_cfr' : 1,   
                        # vector with toxins maximal absorption
                        'k_cfr' : np.array([])          
}

# ------------------------------------------------------------------------------------------------------
# CROSS FEEDING

# nodes dictionary (species)
species_dict_cf = {'A' : 'specialist1',
                   'B' : 'specialist2',
}
# edges dictionary (nutrients)
# for each nutrients who produces it and all the species that use it
nutrients_dict_cf = {'x' : ['AB', 0.4],
                     'y' : ['BA', 0.4]                                        
}
# define toxins dictionary
# for each nutrients who produces it and all the species that use it
toxins_dict_cf = {
}

# make network consumer preferences matrix and metabolic matrix
consumer_preference_matrix_cf, metabolic_matrix_cf, inhibition_matrix_cf, combined_matrix_cf = make_net_mat(species_dict_cf, nutrients_dict_cf, toxins_dict_cf)

def get_matrices_cf():
    return [consumer_preference_matrix_cf, metabolic_matrix_cf, inhibition_matrix_cf, combined_matrix_cf]

# fix parameters
parameters_dict_cf = { # initial resource availability
                        'R_0_cf' : np.array([100,100]),   
                        # initial toxins concentration
                        'T_0_cf' : np.array([]),
                        # initial species abundance                                                
                        'N_0_cf' : np.array([100,100]), 
                        # inverse reinsertion rate                                                  
                        'tau_cf' : np.array([1,1]), 
                        # toxins inverse insetrion rate
                        'tau_t_cf' : np.array([]),
                        # dilution time
                        'tau_dil_cf' : 100,
                        # intrinsic growth rates                                              
                        'g_cf' : np.array([1,0.8]), 
                        # maintainance energy requirements                                                    
                        'm_cf' : np.array([0.1,0.1]),  
                        # external leakage of nutrients                                                         
                        'l_cf' : np.array([0.8, 0.8]),  
                        # toxins leakage
                        'l_t_cf' : np.array([]),
                        # energy content of resources                                               
                        'w_cf' : np.array([1,1]), 
                        # energy content (negative) in toxins i.e. toxicity (?
                        'w_t_cf' : np.array([]),
                        # maximal absorption for Monod function
                        'sig_max_cf' : 1,   
                        # vector with toxins maximal absorption
                        'k_cf' : np.array([])          
}

# ------------------------------------------------------------------------------------------------------
# +- CYCLE

# nodes dictionary (species)
species_dict_pm = {'A' : 'specialist1',
                   'B' : 'specialist2'
}
# edges dictionary (nutrients)
# for each nutrients who produces it and all the species that use it
nutrients_dict_pm = {'x' : ['AB', 0.3],
}
# define toxins dictionary
# for each nutrients who produces it and all the species that use it
toxins_dict_pm = { 
}

# make network consumer preferences matrix and metabolic matrix
consumer_preference_matrix_pm, metabolic_matrix_pm, inhibition_matrix_pm, combined_matrix_pm = make_net_mat(species_dict_pm, nutrients_dict_pm, toxins_dict_pm)

def get_matrices_pm():
    return [consumer_preference_matrix_pm, metabolic_matrix_pm, inhibition_matrix_pm, combined_matrix_pm]

# fix parameters
parameters_dict_pm = { # initial resource availability
                        'R_0_pm' : np.array([100]),   
                        # initial toxins concentration
                        'T_0_pm' : np.array([]),
                        # initial species abundance                                                
                        'N_0_pm' : np.array([1,100]), 
                        # inverse reinsertion rate                                                  
                        'tau_pm' : np.array([1]), 
                        # toxins inverse insetrion rate
                        'tau_t_pm' : np.array([]),
                        # dilution time
                        'tau_dil_pm' : 1000,
                        # intrinsic growth rates                                              
                        'g_pm' : np.array([0.,1]), 
                        # maintainance energy requirements                                                    
                        'm_pm' : np.array([0.,0.5]),  
                        # external leakage of nutrients                                                         
                        'l_pm' : np.array([0.1]),  
                        # toxins leakage
                        'l_t_pm' : np.array([]),
                        # energy content of resources                                               
                        'w_pm' : np.array([1]), 
                        # energy content (negative) in toxins i.e. toxicity (?)
                        'w_t_pm' : np.array([]),
                        # maximal absorption for Monod function
                        'sig_max_pm' : 1,   
                        # vector with toxins maximal absorption
                        'k_pm' : np.array([])
}
