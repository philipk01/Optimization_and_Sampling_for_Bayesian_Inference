
# coding: utf-8

# # *Adaptive* MH
# 
# See the 1999 and 2001 papers of Haario et al.

# In[1]:

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as ss
import math
import random
from collections import namedtuple


from MH_Sampling import acceptance_decision
from FileHandling import save_state
from TestSuite import generate_state_space, generate_iid_samples, get_distribution, get_samples


# In[2]:

# save Adaptive Metropolis parameters in Named Tuple

AM_Pars = namedtuple('AM_Pars', 
                     ['Origin', 'Id',
                      'sigma_0', 'sigma_opt', 
                      'C_0', 'C_opt', 
                      'z_samples'])

# initialize parameters
def init_AM_pars(sp):
    dim, origin, idty, = sp['dim'], sp['Origin'], sp['Id'], 
    sigma_0, sigma_opt = 0.1/np.sqrt(dim), sp['sigma_opt']
    cov_0, cov_opt = sigma_0**2*idty, sigma_opt**2*idty
    return AM_Pars(Origin=origin, Id=idty,
                   sigma_0=sigma_0, sigma_opt=sigma_opt,
                   C_0=cov_0, C_opt=cov_opt, 
                   z_samples=get_samples(sp=sp, name='Z'))


# # Adaptive MH algorithm *AM*

# ## Generate the candidate next sample
# 
# We consider a version of the *Adaptive Metropolis* (*AM*) sampler of Haario
# et al. (2001). We want to sample from the $d$-dimensional target distribution $\pi(\mathbf{x})$. 
# 
# We perform a Metropolis algorithm with covariance matrix $\pmb{Q}_n$ at iteration $n$ given by
# 
# $$\mathbf{Q}_n(\mathbf{x}, ·) = N(\mathbf{x}, \sigma_{0}^2 \mathbb{1}_d)$$
# 
# for $n \leq 2d$, while for $n > 2d$
# 
# $$\mathbf{Q}_n(\mathbf{x}, ·) = (1 − \beta) N(\mathbf{x}, \sigma_{opt}^2 \mathbf{C}_n) + 
# \beta N(\mathbf{x}, \sigma_{0}^2 \mathbb{1}_d)$$
# 
# where $\mathbf{C}_n$ is the current empirical estimate of the covariance of the target distribution
# based on the samples so far, $\sigma_{0}^2 = \frac{0.1^2}{d}$ and $\sigma_{opt}^2 = \frac{2.38^2}{d}$ are the initial and optimal scale, respectively, and $\beta$ is a small positive constant, we use $\beta = 0.05$.
#  
# In other words, the next candidate is sampled from
# 
# $$\mathbf{x}^{*} \sim \mathbf{Q}_n(\mathbf{x}, ·)$$
#  
# The text above is adapted from Section 2 of Gareth O. Roberts and Jeffrey S. Rosenthal (2008) 
# *Examples of Adaptive MCMC*.

# ## Random covariance matrix $M$ from the above paper.

# In[3]:

def get_proposal_cov(M2, n, pars, beta=0.05):
    d, _ = M2.shape
    init_period = 2*d
    s_0, s_opt, C_0 = pars.sigma_0, pars.sigma_opt, pars.C_0
    if np.random.rand()<=beta or n<= init_period:
        return C_0
    else:
        # We can always divide M2 by n-1 since n > init_period
        return (s_opt/(n - 1))*M2


# In[4]:

def generate_AM_candidate(current, M2, n, pars):
    prop_cov = get_proposal_cov(M2, n, pars)
    candidate = ss.multivariate_normal(mean=current, cov=prop_cov).rvs()
    return candidate


# ## Update the mean $\mathbf{m}$ and the the covariance $\mathbf{C}$ 
# 
# In the *AM*-algorithm, the **mean** is updated as
# 
# $$\mathbf{m}_{n+1} = \frac{n}{n+1}\mathbf{m}_{n} + \frac{1}{n+1}\left(\mathbf{x}_{n+1} - \mathbf{m}_{n}\right)$$
# 
# and the **covariance** as
# 
# $$\mathbf{C}_{n+1} = \frac{n}{n+1}\mathbf{C}_{n} + \frac{1}{n+1}\left( 
# \left(\mathbf{x}_{n+1} - \mathbf{m}_{n}\right)\left(\mathbf{x}_{n+1} - \mathbf{m}_{n}\right)^\top - \mathbf{C}_{n} \right)$$
# 
# where $\mathbf{x}_{n+1}$ is the sample generated at step $n+1$.

# In the Welford algorithm, 
# 
# $$M_n \triangleq \sum_{i=1}^{n} {(x_i - \overline{x}_n)}^2$$ 
# 
# or in other words 
# 
# $$s_n^2 = \frac{M_n}{n-1}$$
# 
# It is easier to update $M_n$ in a numerical stable way,
# 
# $$M_n = M_{n-1} + (x_n - \overline{x}_{n+1})(x_n - \overline{x}_n)^\top$$

# In[5]:

def update_moments(mean, M2, sample, n):
    next_n = n + 1
    w = 1/next_n
    new_mean = mean + w*(sample - mean)
    delta_bf, delta_af = sample - mean, sample - new_mean
    new_M2 = M2 + np.outer(delta_bf, delta_af)
    return new_mean, new_M2, next_n


# In[6]:

def multiple_of_10000(n):
    return n%10000 == 0


# In[7]:

def AM_sampler(pars, target, initial_state, run_data):
    ds, N = run_data.DataStore, run_data.N
    target_pdf = target['pdf']
    
    current = initial_state
    mean, M2 = pars.Origin, np.zeros_like(pars.Id)
    accepted = True
    
    for n in range(0, N):
        save_state(data_store=ds, step=n,
                   state=current, value=target_pdf(current),
                   mean=mean, covariance=M2, accepted_p=accepted)
        
        # generate new candidate
        candidate = generate_AM_candidate(current=current, M2=M2, n=n, pars=pars)
        
        # run Metropolis Hastings for acceptance criteria
        accepted = acceptance_decision(current=current, proposed=candidate, pdf=target_pdf)
        
        # accepted candidate becomes new state
        if accepted: 
            current = candidate
        # We always update M2, where S^2 = M2/n-1 
        # whether the proposed samples are accepted or not
        mean, M2, n = update_moments(mean, M2, current, n)
    return run_data

