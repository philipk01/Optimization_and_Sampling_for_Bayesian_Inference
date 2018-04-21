
# coding: utf-8

# # *Metropolis-Hastings* Sampling

# In[ ]:

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as ss
import math
import random
from collections import namedtuple
import numba
#%precision 4
#%matplotlib inline


# In[ ]:

from FileHandling import save_state
from TestSuite import get_samples


# In[ ]:

def acceptance_decision(current, proposed, pdf):
    # Remark: 'accepted_p' includes the case where p_proposed > p_current 
    # since u, a random number between 0 and 1, is then
    # always less than the ratio p_proposed/p_current
    # But for readability we make a distinction between the 
    # between cases below.
    
    p_current, p_proposed = pdf(current), pdf(proposed)
    if p_current <= p_proposed:
        return True
    else:
        u = np.random.rand()
        return u <= p_proposed/p_current


# In[ ]:

# computing the Metroppolis-Hastings acceptance

def compose2(f, g):
    return lambda x: f(g(x))

def likelihood_acceptance_decision(current, proposed, log_pdf):
    # Remark: 'accepted_p' includes the case where p_proposed > p_current 
    # since u, a random number between 0 and 1, is then
    # always less than the ratio p_proposed/p_current
    # But for readability we make a distinction in the code below between the 
    # two cases.
    
    p_current, p_proposed = log_pdf(current), log_pdf(proposed)
    if p_current <= p_proposed:
        return True
    else:
        u = np.random.rand()
        return u <= p_proposed/p_current


# In[ ]:

## Proposal Distribution
# Samples are generated when a run is initialized.

MH_Pars = namedtuple('MH_Pars', ['Proposal'])

def init_MH_pars(sp):
    proposal = sp['Test Suite']['Proposal']
    return MH_Pars(Proposal=proposal)


# In[ ]:

def generate_candidate(center, delta):
    return center + delta


# # Metropolis-Hastings algorithm

# In[ ]:

def MH_sampler(pars, target, initial_state, run_data, C_generation=False, likelihood=True):
    ds, N = run_data.DataStore, run_data.N
    
    target_pdf = target['pdf']
    proposal_samples = pars.Proposal['Samples']

    current = initial_state
    accepted = True
    
    #The integration of the C- and L-variant still has to be done.
    #if C_generation:
    #    generation_function = generate_candidate
    #else:
    #    generation_fuction = L_generate_candidate
        
    if likelihood: 
        decision_function, comparison_function = likelihood_acceptance_decision, compose2(np.log, target['pdf']) 
    else: 
        decision_function, comparison_function = acceptance_decision, target['pdf']
        
    for n in range(1, N):
        save_state(data_store=ds, step=n,
                   state=current, value=target_pdf(current),
                   accepted_p=accepted)
        proposed = generate_candidate(center=current, delta=proposal_samples[n])
        accepted = decision_function(current, proposed, target_pdf)
        if accepted:
            current = proposed
        else:# The else clause is redundant but added for readability.
            current = current
    return run_data


# # Metropolis-Hastings using Cholesky factor $L$ instead of ful covariance matrix $C$
# 

# In[ ]:

def L_generate_candidate(center, L, scale, z_sample):
    return center + scale*L@z_sample


# In[ ]:

def L_MH_sampler(pars, target, initial_state, run_data, likelihood=True):
    ds, N = run_data.DataStore, run_data.N
    sp = target['State Space']
    opt_scale, L = sp['sigma_opt'], sp['Id']
    
    if likelihood: 
        decision_function, comparison_function = likelihood_acceptance_decision, compose2(np.log, target['pdf']) 
    else: 
        decision_function, comparison_function = acceptance_decision, target['pdf']
    
    target_pdf = target['pdf']
    current = initial_state
    accepted = True
    
    z_samples = get_samples(sp=sp, name='Z')
    for n in range(1, N):
        save_state(data_store=ds, step=n,
                   state=current, value=target_pdf(current),
                   accepted_p=accepted)
        proposed = L_generate_candidate(center=current, 
                                        L=L, scale=opt_scale,
                                        z_sample=z_samples[n])
        accepted = decision_function(current, proposed, target_pdf)
        if accepted:
            current = proposed
        else:# The else clause is redundant but added for readability.
            current = current
    return run_data

