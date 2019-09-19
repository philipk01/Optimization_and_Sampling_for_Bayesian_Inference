
# coding: utf-8

# # Save and Inspect the state

# In[66]:


import os
import numpy as np


# In[67]:


def save_state(data_store, step, state, value, accepted_p, 
               mean=None, covariance=None, scale=None, threshold=None, C_evol_pt=None):
    data_store['States'][step] = state
    data_store['Densities'][step] = value
    data_store['Accepted_p'][step] = accepted_p
    data_store['Means'][step] = mean 
    data_store['Covariances'][step] = covariance 
    data_store['Scales'][step] = scale
    data_store['Thresholds'][step] = threshold
    
def inspect_state(data_store, step):
    state = data_store['States'][step]
    value = data_store['Densities'][step] 
    accepted_p = data_store['Accepted_p'][step] 
    mean = data_store['Means'][step] 
    covariance = data_store['Covariances'][step] 
    scale = data_store['Scales'][step]
    threshold = data_store['Thresholds'][step] 
    print("State:", state, "R: ", scale, "\nThreshold: ", threshold, "\nState: ", state, 
          "\nIt's value: ", value, "\nMean: ", mean, 
          "\nCovariance: ", covariance)


# # Save in the file format used by *PyMC3*

# ## Structure of the data directory
#  
# The directory structure of 'Data' is as follows
#  
#     1. for each dimension $d$ of the state space, 'Data' contains a folder 'Dim d'  
#     2. for each target, 'Dim d' contains a folder 'Target k' where $k$ is the index of 
#     that targets in the test suite
#     3. for each sampler, 'Target k' contains a folder named after that sampler
#     4. for each run given the dimension of the state space, the target and the sampler, 
#     a file 'chain_i' is generated where $i$ is the index of the run. 
#     
# 
# The global variable PARENT_FOLDER contains the parent folder, i.e. the folder where the experimental data will be store, e.g.
# 
#     PARENT_FOLDER = '/Users/BM/Documents/Programming/Python/Notebooks/MCMC/To execute a run'
#     
# The functions below assume that the parent folder is correctly set.

# In[68]:


def relative_path_to_chain(dim, t_name, s_name):
    data_folder = 'Data'
    dim_folder = 'Dimension_{}'.format(dim)
    target_folder = t_name
    sampler_folder = s_name
    return './'+'/'.join([data_folder, dim_folder, target_folder, sampler_folder])

class ChDir(object):
    """
    Step into a directory temporarily.
    """
    def __init__(self, path):
        self.old_dir = os.getcwd()
        self.new_dir = path
 
    def __enter__(self):
        os.chdir(self.new_dir)
 
    def __exit__(self, *args):
        os.chdir(self.old_dir)

def save_chain(chain, idx, individual_components_p=True):
    """Save a single-chain trace with index 'idx'. PyMC3 uses the labels x__0, x__1, x__2, etc.
    for a vector when are regarded as COMPONENTS of that vector. 
    If we want to treat them INDIVIDUALLY the labels x_0, x_1, x_2, etc. have to be used. 
    This is, we use double versus single underscore.
    """
    chain_name = 'chain-{}.csv'.format(idx)
    _, nbcols = chain.shape
    underscore = '_' if individual_components_p else '__'
    varnames = ['x{}{}'.format(underscore, index) for index in range(nbcols)]
    header = ','.join(varnames)
    np.savetxt(fname=chain_name, X=chain, header=header, comments='', delimiter=',')

def save_run_data(run_data, parent_folder):
    warning = 'Parent Folder \'%s\' does NOT exist'%(parent_folder)
    if not os.path.exists(parent_folder):
        return warning
    chain = run_data.DataStore['States']
    chain_folder = relative_path_to_chain(dim=run_data.StateSpace['dim'],
                                          t_name=run_data.Target['Name'] , 
                                          s_name=run_data.Sampler['Name'])
    if not os.path.exists(chain_folder):
        os.makedirs(chain_folder)
    with ChDir(chain_folder):
        nbfiles = len(os.listdir())
        save_chain(chain=chain, idx=nbfiles)

def save_comparison(combined_data, parent_folder):
    for i, run_data in enumerate(combined_data):
        save_run_data(run_data, parent_folder)


# In[75]:


def read_states(f_name, dim, t_name, s_name):
    chains_folder = relative_path_to_chain(dim=dim, t_name=t_name, s_name=s_name)
    with ChDir(chains_folder):
        return np.loadtxt(fname=f_name, skiprows=1, delimiter=',')

