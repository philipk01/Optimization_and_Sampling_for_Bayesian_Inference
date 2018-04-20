
# coding: utf-8

# # Gaussian Adaptation *Sampling*

# In[8]:

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as ss
import math
import random
from collections import namedtuple


# In[3]:

from MH_Sampling import acceptance_decision
from FileHandling import save_state
from TestSuite import generate_state_space, generate_iid_samples, get_distribution, get_samples


# # *Maximum Entropy* Principle
# To be done.
# 
# The entropy of the multivariate normal distribution $N(\mathbf{m}, \mathbf{C})$ with mean $\mathbf{m}$ and covariance matrix $\mathbf{C}$ is 
# 
# $$ H(N) = \ln \sqrt{(2 \pi e)^n \det \mathbf{C}}$$

# In[4]:

def entropy(cov):
    dim1, dim2 = cov.shape
    assert dim1==dim2
    return math.log(np.sqrt((2* math.pi * math.e)**dim1 * la.det(cov)))


# ## Check Covariance Matrix

# In[5]:

def analyse_cov(cov):
    eigenvals, eigenvecs = la.eig(cov)
    print('Covariance Matrix:\n', cov)
    print('Determinant:', la.det(cov))
    print('Eigenvalues:', eigenvals)
    print('Eigenvectors:', eigenvecs)
    print('Symmetric:', np.allclose(cov, cov.T))


# # Gaussian Adaptation according to Mueller's Matlab code
# 
# This notebook is based on Mueller's Matlab code and the paper *"Gaussian Adaptation as a unifying framework for black-box optimization and adaptive Monte Carlo sampling"* by *Christian I. Muellen* and *Ivo F. Sbalzarini*.

# ## Strategy Parameters
# 
# The **strategy parameters** are:
# - the **acceptance ratio** $P$
# - the **expansion** $f_e > 1$ and **contracton factor** $f_c < 1$ used to update the global scale $\sigma$
# - the **weigths** $\lambda_{\mathbf{m}}$, $\lambda_{\mathbf{C}}$, and $\lambda_{\theta}$ 
# used to update the mean $\mathbf{m}$, the covariance matrix $\mathbf{C}$, and the threshold $\theta$, respectively.
# 
# They are **initialized** as follows, cf. p.2 of the MATLAB code of Mueller:
# 
# - **acceptance ratio** $P = \frac{1}{e}$. **REMARK BM**: Check the explanation regarding $P = \frac{s}{s+f}$ where $s$ and $f$ are the number of successes and failures so far.
# 
# - **expansion factor** $f_e = 1 + \beta (1 - P)$ and **contraction factor** 
# $f_c = 1 - \beta P$ where $\beta = \lambda_{\mathbf{C}}$
# 
# - **weights** are initialized as follows
# 
# - $\lambda_{\mathbf{C}} = \frac{\ln (d+1)}{(d+1)^2}$
# - $\lambda_{\mathbf{m}} = \frac{1}{ed}$
# - $\lambda_{\theta} = \frac{1}{ed}$ without restart, cf. the end of Section II.B of the paper what to do in case of restart.
# 
# Here, $d$ is the dimension of the **search space** in case of **optimization** or the **state space**
# in case of **sampling**.

# ### Initializing *strategy parameters*
# Cf. above for their initial values.

# In[30]:

GaA_Pars = namedtuple('GaA_Pars', 
                      ['l_C', 'l_m', 'b', 'P', 
                       'f_e', 'f_c', 'max_scale', 'max_cond', 
                       'Origin', 'Id'])


# In[6]:

def init_GaA_pars(sp):
    D, origin, identity = sp['dim'], sp['Origin'], sp['Id']
    tmp_l_c = math.log(D+1)/(D + 1)**2
    tmp_P = 1/math.e
    return GaA_Pars(l_C=tmp_l_c, 
                    l_m=1/(math.e*D), 
                    b=tmp_l_c, 
                    P=tmp_P, 
                    f_e=1 + tmp_l_c*(1-tmp_P), 
                    f_c=1 - tmp_l_c*tmp_P, 
                    max_scale=1000,
                    max_cond=80, # he value used by Mueller is 1e6*D but this results in errors
                    Origin=origin,
                    Id=identity)


# In[10]:

def display_parameters(pars):
    str_1 = "l_C: {:1.4f}\nl_m: {:1.4f}\nb: {:1.4f}\nP: {:1.4f}"
    str_2 = "\nf_e: {:1.4f}\nf_c: {:1.4f}\nmax_scale: {:1.4f}\nmax_cond: {:1.4f}"
    pars_info_1 = str_1.format(pars.l_C, pars.l_m, pars.b, pars.P)
    pars_info_2 = str_2.format(pars.f_e, pars.f_c, pars.max_scale, pars.max_cond)
    print( pars_info_1,  pars_info_2)


# # Generate next sample using $\mathbf{Q}$
# 
# The new state $\mathbf{x}_{n+1}$ is generated as follows
# 
# $$\mathbf{x}_{n+1} = \mathbf{m}_{n} + \sigma_n \mathbf{Q}_{n} \mathbf{z}_{n}$$ 
# 
# where $\sigma_n$ is the global scale, $\mathbf{Q}_{n}$ is the "square root" of the covariance matrix $\mathbf{C}_{n}$ as defined below, and 
# $\mathbf{z}_{n}$ is a sample of the multivariate standard normal distribution $N(0,1)$ generated at step $n$.

# In[11]:

def Q_generate_GaA_candidate(mean, scale, Q, z_sample):
    # This function uses the normalized sqrt Q of the covariance matrix C. 
    # cf. p.7 of the MATLAB code of Mueller
    x = mean + scale*(Q @ z_sample.T)
    return x


# # Generate next sample using $C$

# In[12]:

def C_generate_GaA_candidate(mean, C, z_sample):
    return mean + C @ z_sample.T


# # Updates of the *scale* $\sigma$, the *mean* $\mathbf{m}$, and the *threshold* $\theta$

# ## Update of the **scale** $\sigma$
# 
# The **scale** is *updated at each step*: 
# 
# When the new sample is **accepted** then the scale is **increased**:
# 
# $$\sigma_{n+1} = f_e \sigma_n$$
# where $f_e > 1$ is the *expansion factor*, one of the strategy parameters of Gaussion Adaptation. 
# 
# When the sample is **rejected** then the scale is **decreased**:
# $$\sigma_{n+1} = f_c \sigma_n$$
# where $f_c < 1$ is the *contraction factor*, another strategy parameter. 

# In[13]:

def contract(scale, pars):
    return pars.f_c*scale

def expand(scale, pars):
    # cf. p.10 of the MATLAB code of Mueller
    f_e, max_scale = pars.f_e, pars.max_scale
    next_scale = f_e*scale
    if next_scale <= max_scale:
        return next_scale
    else: 
        return max_scale


# ## Update of the *mean* $\mathbf{m}$
# 
# 
# These are 
# 
# The **mean** is **only updated** when the new sample $\mathbf{x}_{n+1}$ is **accepted**. Th new mean is
# 
# $$\mathbf{m}_{n+1} = (1 - \lambda_{\mathbf{m}}) \mathbf{m}_{n} + \lambda_{\mathbf{m}} \mathbf{x}_n$$
# 
# Here, $\lambda_{\mathbf{m}}$ and $\lambda_{\mathbf{C}}$ are *strategy parameters* of *Gaussian Adaptation*.

# In[14]:

# This code is for global optimization, NOT sampling. 
def GaA_mean_update_2(mean, sample, pars):
    l_m = pars.l_m
    return (1-l_m)*mean + l_m*sample


# In[15]:

# In case of sampling l_m = 1, in other words the new sample becomes the next mean.
def GaA_mean_update(mean, sample, pars):
    return sample


# # Update of the covariance matrices $\mathbf{C}$ and $\mathbf{Q}$

# $\mathbf{C}$ and $\mathbf{Q}$ are covariance matrices and therefore positive definite and symmetric. 
# Symmetry might get lost due to rounding off errors in the update process. After each update
# we make sure that the result is still symmetric.
# 
# The first way to do this uses the *Numpy*-function *triu* that returns the upper triangle part of a matrix.
# The second one uses the *transpose* of a matrix. Recall that $\mathbf{S} = \mathbf{S}^\top$ for a symmetric matrix
# $\mathbf{S}$.

# In[16]:

def trui_enforce_symmetry(cov):
    dim1, dim2 = cov.shape
    assert dim1==dim2
    return np.triu(cov,0)+np.triu(cov,1).T

def transpose_enforce_symmetry(cov):
    dim1, dim2 = cov.shape
    assert dim1==dim2
    return 1/2*(cov+cov.T)


# # Update of the "*square root*" $\mathbf{Q}$ of the *covariance* matrix $\mathbf{C}$

# First, we calculate $\Delta \mathbf{C}_n$ as follows
# 
# $$\Delta \mathbf{C}_{n+1} = (1-\lambda_\mathbf{C})\mathbb{1}_d + \lambda_\mathbf{C} \mathbf{z}_n \mathbf{z}_n^\top$$
# 
# where $\mathbb{1}_d$ is the identity matrix, $\mathbf{z}_n$ is the $n$th sample of the multivariate standard Gaussian distribution, and $\lambda_\mathbf{C}$ is the strategy parameter used in the update of the covariance matrix $\mathbf{C}$.

# In[17]:

def delta_C(z_sample, pars):
    l_C = pars.l_C
    identity = pars.Id
    deltaC = (1-l_C)*identity + l_C*np.outer(z_sample, z_sample)
    #return enforce_symmetry(deltaC)
    return deltaC


# Next, we define $\Delta \mathbf{Q}_{n+1}$ as 
# 
# $$\Delta \mathbf{Q}_{n+1} \triangleq \sqrt{\Delta \mathbf{C}_{n+1}}$$

# In[18]:

def sqrtm(cov):    
    D, B = la.eigh(cov)
    sqrtD = np.diag(np.sqrt(D))
    # Return the sqrt Q of the matrix C
    return B @ sqrtD @ B.T


# Finally, we calculate $\mathbf{Q}_{n+1}$ as
# $$\mathbf{Q}_{n+1} = \mathbf{Q}_n \Delta \mathbf{Q}_{n+1}$$

# In[19]:

def normalize(cov):
    D, _ = cov.shape
    normalization_constant = la.det(cov)**(1/D)
    normalized_cov = cov/normalization_constant
    #det = la.det(normalized_cov)
    #np.testing.assert_almost_equal(det, 1.0)
    return normalized_cov


# In[20]:

def GaA_Q_update(z_sample, Q, pars):
    max_cond = pars.max_cond
    deltaC = delta_C(z_sample, pars)
    deltaQ = sqrtm(deltaC)
    Q_next = normalize(transpose_enforce_symmetry(Q @ deltaQ))
    if la.cond(Q_next) <=  max_cond:
        return Q_next
    else: 
        return Q


# # Update of the *covariance* matrix $\mathbf{C}$

# In[21]:

def GaA_C_update(C, mean, sample, pars):
    # Cf. p.10 of the MATLAB code of Mueller
    l_C, max_cond = pars.l_C, pars.max_cond
    delta = mean - sample
    C_next = (1 - l_C)*C + l_C*np.outer(delta, delta)
    if la.cond(C_next) <= max_cond:
        return C_next
    else: 
        return C  


# # Gaussian Adaptation Sampling

# In[22]:

def Q_GaA_sampler(pars, target, initial_state, run_data):
    target_pdf, sp = target['pdf'], target['State Space']
    Origin, Id = sp['Origin'], sp['Id']
    
    ds, N = run_data.DataStore, run_data.N
    z_samples = get_samples(sp=sp, name='Z')
    
    #Set up and save the initial state
    m = x_current = initial_state
    sigma = 1
    Q = Id
    save_state(data_store=ds, 
               step=0, 
               state=x_current, 
               value=target_pdf(x_current),
               accepted_p=True, 
               mean=m, 
               covariance=Q, 
               scale=sigma, 
               threshold=None)
    
    #Sample and save state
    for n in range(1, N):
        z_sample = z_samples[n]
        x_proposed = Q_generate_GaA_candidate(mean=x_current, 
                                              scale=sigma, 
                                              Q=Q, 
                                              z_sample=z_sample)
        accepted = acceptance_decision(x_current, x_proposed, target_pdf)
        if accepted:
            x_current = x_proposed
            sigma = expand(sigma, pars=pars)
            m = GaA_mean_update(mean=m, sample=x_proposed, pars=pars)
            Q = GaA_Q_update(Q=Q, z_sample=z_sample, pars=pars)
        else:
            sigma = contract(sigma, pars=pars)
        save_state(data_store=ds, 
                   step=n, 
                   state=x_current, 
                   value=target_pdf(x_current),
                   accepted_p=accepted, 
                   mean=m, 
                   covariance=Q, 
                   scale=sigma, 
                   threshold=None)
    return run_data


# In[23]:

def C_GaA_sampler(pars, target, initial_state, run_data):
    target_pdf, sp = target['pdf'], target['State Space']
    Origin, Id = sp['Origin'], sp['Id']
    
    ds, N = run_data.DataStore, run_data.N
    z_samples = get_samples(sp=sp, name='Z')

    #Set up and save the initial state
    m = x_current = initial_state
    sigma = 1
    C = Id
    
    save_state(data_store=ds, 
               step=0, 
               state=x_current, 
               value=target_pdf(x_current),
               accepted_p=True, 
               mean=m, 
               covariance=C, 
               scale=sigma, 
               threshold=None)
    
    #Sample and save state
    for n in range(1, N):
        z_sample = z_samples[n]
        x_proposed = C_generate_GaA_candidate(mean=x_current,
                                              C=C, 
                                              z_sample=z_sample)
        accepted = acceptance_decision(x_current, x_proposed, target_pdf)
        if accepted:
            x_current = x_proposed
            sigma = expand(sigma, pars=pars)
            m = GaA_mean_update(mean=m, sample=x_proposed, pars=pars)
            C = GaA_C_update(C=C, mean=m, sample=x_proposed, pars=pars)
        else:
            sigma = contract(sigma, pars=pars)
        save_state(data_store=ds, 
                   step=n, 
                   state=x_current, 
                   value=target_pdf(x_current),
                   accepted_p=accepted, 
                   mean=m, 
                   covariance=C, 
                   scale=sigma, 
                   threshold=None)
    return run_data

