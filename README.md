## Simple non spatial model for microbial communities

# Dependencies

The model requires the following libraries:

* pandas
* numpy
* scipy
* matplotlib
* networkx
* PIL

# Structure

* *__init__.py* is the main file, where all other modules are imported and the simulations can be run, here the simulation time is set and a network is chosen from the ones defined in the *specifications.py* file; if one prefers to start from the matrices instead of the networks, pre-created matrices can be loaded in this step, otherwise they are created from the selected network; after defining the initial state vector, the model is thus run
* *models.py* is the file containing the function for running the model anplotting the populations and nutrients dynamics
* *aux.py* is the file containing all the auxiliary functions the model uses, sucas the uptake function, the replenishment function and the dynamics of chemicals anpopulations
* *specifications.py* is the file where the networks are defined and thparameters fixed; a Community class is defined to facilitate the running of the model; thnetworks are defined through 3 dictionaries: the species dictionary (keys=lettersvalues=name of species), the nutrients dictionary (keys=nutrient, values=couples of nodes iconnects ond strength of the connection i.e. uptake rate) and the toxins dictionar(keys=toxin, values=couples of nodes it connects ond strength of the connection i.e. uptakrate)
* *make_network.py* is the file where functions to create network and matricestarting from the dictionaries is defined
