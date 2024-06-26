"""
Main code for running the model, here a network from specifications.py is loaded and the simulation is run until steady state of population

"""

from re import S
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from aux import *
from models import *
from make_network import *
from specifications import *
import specifications
import scipy
import importlib
import sys

importlib.reload(specifications)


# prerape parameters dictionary
# parameters list of keys
keys_list = ['R_0','T_0','N_0','tau','g','m','l','l_t','w','w_t']

param_dict_specific = specifications.parameters_dict_pm
param_dict = {}
for old_key, new_key in zip(param_dict_specific.keys(), keys_list):
        param_dict[new_key] = param_dict_specific[old_key]


# prepare matrices or import them
matrices = get_matrices_pm()
# matrices file paths
# cp_mat, inh_mat, comb_mat, met_mat = 

# prepare initial state
y_0 = [param_dict['N_0'],param_dict['R_0'],param_dict['T_0']]

# initialize community
community = Community(y_0,[dNdt,dR_ss,dR_ss_sep,dT_ss_sep],param_dict,matrices)

# run model
community.run_steady_state_sep()


