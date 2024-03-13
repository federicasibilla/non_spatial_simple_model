"""
main code for running the model, here a community class is defined and the parameters are set

"""

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
keys_list = ['R_0','T_0','N_0','tau','tau_t','tau_dil','g','m','l','l_t','w','w_t','sig_max','k','t_N']

param_dict_specific = specifications.parameters_dict_pm
param_dict = {}
for old_key, new_key in zip(param_dict_specific.keys(), keys_list):
        param_dict[new_key] = param_dict_specific[old_key]

# prepare matrices
matrices = get_matrices_pm()

# prepare initial state
y_0 = [param_dict['N_0'],param_dict['R_0'],param_dict['T_0']]

# initialize community
pm_community = Community(y_0,[dNdt,dR_ss],param_dict,matrices)

# run model
pm_community.run_steady_state()

