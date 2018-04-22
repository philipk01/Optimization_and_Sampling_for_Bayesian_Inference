
# coding: utf-8

# # *Adaptive* MH using *Cholesky decomposition* of the covariance
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
import numba
from collections import namedtuple
get_ipython().run_line_magic('precision', '4')
get_ipython().run_line_magic('matplotlib', 'inline')


# from importlib import reload 
# reload(ut)

# In[2]:

from MH_Sampling import acceptance_decision
from FileHandling import save_state
from TestSuite import generate_state_space, generate_iid_samples, get_samples


# In[3]:

AM_Pars = namedtuple('AM_Pars', 
                     ['Origin', 'Id',
                      'sigma_0', 'sigma_opt', 'L_0', 
                      'z_samples'])

def init_AM_pars(sp):
    dim, origin, idty = sp['dim'], sp['Origin'], sp['Id'] 
    sigma_0, sigma_opt = 0.1/np.sqrt(dim), sp['sigma_opt']
    L_0 = idty
    return AM_Pars(Origin=origin, Id=idty,
                   sigma_0=sigma_0, sigma_opt=sigma_opt, L_0=L_0,
                   z_samples=get_standard_normal_samples(sp))


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

# In[4]:

Factors = namedtuple('Factors',
                     ['Chol', 'Scale'])


# In[5]:

def get_prop_data(L, n, pars):
    beta = 0.05
    d, _ = L.shape
    sigma_0, sigma_opt, L_0 = pars.sigma_0, pars.sigma_opt, pars.L_0
    init, current = Factors(Chol=L_0, Scale=sigma_0), Factors(Chol=L, Scale=sigma_opt)
    init_period = 2*d
    if n <= init_period:
        return init
    else:
        return current if np.random.binomial(n=1, p=1-beta) else init


# # Generation of candidate
# 
# If the proposal distribution is the $d$-dimensional multivariate normal distribution $N(\pmb{m}, \pmb{C})$ then 
# the next candidate $\pmb{x}^{*}$ is generated according to that distribution, i.e. 
# 
# $$\pmb{x}^{*} \sim N(\pmb{m}, \pmb{C})$$
# 
# If $L$ is the lower Cholesky factor of $C$, i.e. $C = L L^\top$ this can be rewritten as 
# 
# $$\pmb{x}^{*} = \pmb{m} + L \pmb{z}$$
# 
# where $\pmb{z} \sim N(\pmb{0}, \mathbb{1}_d)$ is a sample of the $d$-dimensional standard normal distribution. 
# 
# In case of $$\pmb{x}^{*} \sim N(\pmb{m}, \sigma^2 \pmb{C})$$ this becomes
# 
# $$\pmb{x}^{*} = \pmb{m} + \sigma L \pmb{z}$$
# 
# 

# In[6]:

def C_generate_candidate(m, C, s):
    return 


# In[7]:

def L_generate_candidate(m, L, s, z):
    return m + s*L@z


# In[8]:

# see "A More Efficient Rank-one Covariance Matrix Update for Evolution Strategies" Igel, Krause 2015
# and adapted slightly to incoporate alpha, beta != 1
@numba.jit(nopython=True)
def rank_1_update(L, u, alpha, beta):
    assert alpha > 0, 'Argument alpha should be positive'
    assert beta > 0, 'Argument beta should be positive'
    d = len(u)
    L = np.sqrt(alpha)*L  #Added
    b = 1
    nL = np.zeros_like(L)
    v = np.copy(u)  #Added
    for j in np.arange(d):
        nL[j,j] = np.sqrt(L[j,j]**2 + (beta/b)*(v[j]**2))
        gamma = b*L[j,j]**2 + beta*v[j]**2
        for k in range(j+1, d):
            v[k] = v[k] - (v[j]/L[j,j])*L[k,j]
            nL[k,j] = (nL[j,j]/L[j,j])*L[k,j] + (nL[j,j]*beta*v[j]/gamma)*v[k]
        b = b + beta*(v[j]**2/L[j,j]**2)
    return nL


# In[9]:

def update_moments(mean, L, sample, n):
    next_n = n + 1
    w = 1/next_n
    new_mean = mean + w*(sample - mean)
    new_L = rank_1_update(L=L, u=sample, alpha=1-w, beta=w)
    return new_mean, new_L, next_n


# In[10]:

@numba.jit
def update_L(samples):
    N, d = samples.shape
    initial_period = 2*d
    initial_cov = np.cov(samples[:initial_period], rowvar=False)
    initial_mean = np.mean(samples[:initial_period], axis=0)
    C = initial_cov
    L = la.cholesky(initial_cov) 
    mean = initial_mean
    for n in range(initial_period, len(samples)):
        sample = samples[n]
        w = 1/(n+1)
        L = rank_1_update(L, sample-mean, alpha=(n-1)/n, beta=w)
        mean = (1-w)*mean + w*sample
    return L@L.T


# In[11]:

def AM_sampler(pars, target, initial_state, run_data): 
    ds, N = run_data.DataStore, run_data.N
    
    
    target_pdf = target['pdf']
    z_samples = pars.z_samples
    
    current = initial_state
    mean, L, sigma_0 = pars.Origin, pars.L_0, pars.sigma_0
    accepted = True
    d = len(initial_state)
    init_period = 2*d
    samples=[]
    for n in range(init_period):
        save_state(data_store=ds, step=n,
                   state=current, value=target_pdf(current),
                   mean=mean, covariance=L, accepted_p=accepted)
        candidate = L_generate_candidate(m=current, L=L, s=sigma_0, z=z_samples[n])
        accepted = MH_decision(current=current, proposed=candidate, pdf=target_pdf)
        if accepted: 
            current = candidate
        else:
            current = current
        samples.append(current)
    # Calculate the first two moments at the end of initial period.
    initial_cov = np.cov(samples, rowvar=False)
    initial_mean = np.mean(samples, axis=0)
    C = initial_cov
    L = la.cholesky(initial_cov) 
    mean = initial_mean
    
   
    # Once the initial period is finished we start to adapt.
    for n in range(init_period, N):
        #if n%1000 == 0:
        #    print('n:', n)
        save_state(data_store=ds, step=n,
                   state=current, value=target_pdf(current),
                   mean=mean, covariance=L, accepted_p=accepted)
        
        p_L, p_sigma = get_prop_data(L=L, n=n, pars=pars)
        candidate = L_generate_candidate(m=current, L=p_L, s=p_sigma, z=z_samples[n])
        accepted = MH_decision(current=current, proposed=candidate, pdf=target_pdf)
        if accepted: 
            current = candidate
        mean, L, n = update_moments(mean, L, current, n)
    return run_data

