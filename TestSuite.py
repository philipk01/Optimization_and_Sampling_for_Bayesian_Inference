
# coding: utf-8

# In[3]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as ss
import math
import random
from collections import namedtuple
#get_ipython().magic('precision 4')
#get_ipython().magic('matplotlib inline')


# # State Space
# The dictionary *State Space* contains its dimension, the origin and identity matrix,
# and the optimal $\sigma_{opt}$ scale according to Gelman et al.: $\sigma_{opt} = 2.38/\sqrt{d}$,
# where $d$ is the dimension of the state space and $\sigma_{opt}$ is the scale of the isotropic proposal distribution used in the Metropolis-Hastings sampler.
# 
# Later on we will add the target distributions of the test suite used in our experiments.

# In[2]:


def calculate_optimal_sigma(dim):
    return 2.38/np.sqrt(dim)

def state_space(dim):
    return {'dim': dim, 'Origin': np.zeros(dim), 'Id': np.eye(dim), 
            'sigma_opt':calculate_optimal_sigma(dim)}


# ## A random state of the state space

# In[3]:


def generate_random_state(sp, min_range=-10, max_range=10):
    """Generates a random state in the state space that fits in the area to be plotted.
    """
    return np.random.uniform(low=min_range, high=max_range, size=sp['dim'])


# In[4]:


#random.seed(10) #REMARK BM: Random seed doesn't seem to work.


# In[5]:


def generate_initial_states(sp, nb_runs):
    initial_states = {i:generate_random_state(sp) for i in np.arange(nb_runs)}
    # Only update if the key does not exist yet. Check out how to do this.
    sp.update({'Initial States':initial_states})


# # Testsuite of Target Distributions
# 
# ## Uncorrelated and Correlated Gaussian Distributions
# 
# $\pi_1$ is the uncorrelated Gaussian distribution with covariance matrix
# 
# $$
# C_u=
#   \begin{pmatrix}
#     100 & 0 \\
#     0 & 1 
#   \end{pmatrix}
# $$
# 
# and $\pi_2$ is the correlated Gaussion distribution with 
# covariance matrix
# 
# $$
# C_c=
#   \begin{pmatrix}
#     50.5 & 49.5 \\
#     49.5 & 50.5 
#   \end{pmatrix}
# $$

# # Covariance Matrix

# In[6]:


def generate_rotation_matrix(theta):
    # Rotation matrix is 2-dimensional
    return np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])

def generate_correlated_cov(uncorrelated_cov, theta):
    correlated_cov = np.copy(uncorrelated_cov)
    R = generate_rotation_matrix(theta)
    R_inv = la.inv(R)
    # Rotate the first 2 dimensions only and leave the other dimensions
    # of the covariance matrix intact.
    correlated_cov[:2, :2] = R @ uncorrelated_cov[:2,:2] @ R_inv
    return correlated_cov


# ### We could also use the fact that the transpose of a rotation is also its inverse.

# def alt_generate_correlated_cov(uncorrelated_cov, theta):
#     # Here we use the fact that the transpose of a rotation is also its inverse.
#     correlated_cov = np.copy(uncorrelated_cov)
#     R = generate_rotation_matrix(theta)
#     correlated_cov[:2, :2] = R @ uncorrelated_cov[:2,:2] @ R.T
#     return correlated_cov

# ## Contour Functions corresponding with the Target Distributions 
# 
# ### Standard Ellipse and General Ellipse
# 
# When the ellips with equation 
# 
# $$\left( \frac{x_1}{a} \right)^2 + \left( \frac{x_2}{b} \right)^2 = 1$$
# 
# is rotated over an angle $\theta$ then the equation of that ellips becomes
# 
# $$\underbrace{\left(\frac{\cos^2\theta}{a^2} + \frac{\sin^2\theta}{b^2}\right)}_\text{A} x_1^ 2 + 
# \underbrace{\left( \frac{\sin^2\theta}{a^2} + \frac{\cos^2\theta}{b^2} \right)}_\text{C} x_2^2  
# + 2 \underbrace{\cos \theta \sin \theta \left( \frac{1}{a^2} - \frac{1}{b^2} \right)}_\text{B} x_1 x_2 = 1$$
# 
# or 
# 
# $$A x_1^2 + 2 B x_1 x_2 + C x_2^2 = 1$$
# 
# where 
# 
# $$B^2 - A C < 0$$
# actually 
# 
# $$B^2 - A C = -1/(ab)^2$$

# In[7]:


def get_ellipse_parameters(cov):
    
    """Get the first 2 eigenvalues and their angle of covariance matrix.
    The eigenvalues are returned in descending order together with 
    the angle of rotation (in radians). The eigenvalues correspond with 
    half the length, a and b, of these two main axes of 
    the general ellipse.
    If the angle is small enough, meaning that the covariance matrix 
    can be considered diagonal, 0.0 is returned."""
    
    e, v = la.eig(cov)
    e_1, e_2, *_ = e
    a, b = np.sqrt(e_1), np.sqrt(e_2)
    v_a, v_b, *_ = v
    # a must be at least b
    if a < b:
        a, b = b, a
        v_a, v_b = v_b, v_a   
    cos, *_ = v_a
    theta = np.arccos(cos)
    if np.isclose(theta, 0):
        theta = 0.0
    return a, b, theta


# In[8]:


def calculate_ellipse_coefficients(a, b, theta):
    sin, cos = np.sin(theta), np.cos(theta)
    cos_sqd, sin_sqd = cos**2, sin**2
    a_sqd, b_sqd = a**2, b**2
    A = cos_sqd/a_sqd + sin_sqd/b_sqd
    C = sin_sqd/a_sqd + cos_sqd/b_sqd
    B = (1/a_sqd - 1/b_sqd)*sin*cos
    return A, B, C


# In[9]:


def get_Gaussian_contour(cov):
    a, b, theta = get_ellipse_parameters(cov)
    A, B, C = calculate_ellipse_coefficients(a, b, theta)
    return lambda x1, x2: A*x1**2 + 2*B*x1*x2 + C*x2**2


# # Distribution
# 
# We have three kind of distributions in the test suite
#    1. Gaussian distributions
#    2. mixture of Gaussians
#    3. transformed Gaussians, the so called twist distributions
#    
# The second kind is not implemented yet.
# 
# 
# The dictionary *Gaussian* contains the following fields
#    * its *Name*
#    * the *State Space* on which the probability distribution is defined
#    * its *probabibility density function* or *pdf*
#    * *Samples* that are *independent and identically distributed*. These samples will be compared to
#    the samples generated by the MCMC samplers studied. These samples are added at run time.
#    * the *Contour Function* used to plot the 
#    * *Contour Levels* corresponding to the preset confidence levels, cfr. the global variable 
#    CONFIDENCE_LEVELS for the values used. The values of the $\chi^2$ distribution corresponding to
#    the confidence levels used: 67, 90, 95 and 99 percent.
#    
# Additionarly to the fields of Gaussian dictionary, *non_Gaussian* contains the additional fields *Transformation*,
# this is the function that will generated its i.i.d. samples using the samples of generating Gaussian.

# ## Gaussian Distributions in the Test Suite

# ### Draw the contour lines corresponding to preset *confidence levels*

# In[1]:


def get_chi2s(df, confidence_levels=[0.67, 0.90, 0.95, 0.99]):
    """ppf stands for the percent point function (inverse of cdf — percentiles)."""
    #contour_levels = {conf:ss.chi2.ppf(conf, df) for conf in confidence_levels}
    contour_levels = [ss.chi2.ppf(conf, df) for conf in confidence_levels]
    return contour_levels


# ### Generate the Gaussians given their covariances

# In[11]:


def generate_Gaussian(sp, name, mean, cov):
    d = sp['dim']
    rv = ss.multivariate_normal(mean=mean, cov=cov)
    return {'Name':name,
            'State Space':sp,
            'pdf':rv.pdf, 
            'Mean':mean,
            'Covariance':cov,
            'Contour Function':get_Gaussian_contour(cov),
            'Contour Levels':get_chi2s(df=2)
            #'Samples':None,
           }


# In[12]:


def generate_covs(sp):
    # Standard Normal Z has the identity matrix as covariance
    identity = sp['Id']
    
    # The optimal isotropic proposal is $\sigma_{opt} * Id$
    var_opt = sp['sigma_opt']**2
    prop_cov = var_opt*identity
    
    # P1_2
    Pi_1_cov = np.copy(identity)
    Pi_1_cov[0, 0] = 100
    
    # Pi_2
    Pi_2_cov = generate_correlated_cov(Pi_1_cov, np.pi/4)
    
    # Pi_rnd
    d = sp['dim']
    M = np.random.normal(size=(d,d))
    Pi_rnd = M@M.T
    return {'Z':identity, 'Proposal':prop_cov, 'Pi_1':Pi_1_cov, 
            'Pi_2':Pi_2_cov, 'Pi_rnd':Pi_rnd}
    

def generate_all_Gaussians(sp):
    named_covs = generate_covs(sp)
    gaussians = {name:generate_Gaussian(sp=sp, name=name, mean=sp['Origin'], cov=cov)
                 for name, cov in named_covs.items()}
    return gaussians


# ## Proposal Generator
# 
# The **radial basis** or **isotropic** proposal generator used by the Metropolis-Hastings sampler. 
# Its *mean* is the origin and the *spread* is $\sigma$.

# In[13]:


def generate_isotropic_Gaussian(sp, sigma):
    origin, identity = sp['Origin'], sp['Id']
    diagonal = sigma**2 * identity
    return generate_Gaussian(sp=sp, name='Isotropic', mean=origin, cov=diagonal)


# In[14]:


def generate_random_Gaussian(sp):
    d, origin = sp['dim'], sp['Origin']
    M = np.random.normal(size=(d,d))
    random_cov = M@M.T
    return generate_Gaussian(sp=sp, name='Random', mean=origin, cov=random_cov)


# ## Twisted Distributions in the Test Suite

# In[15]:


def f_twist(b):
    def phi_b(x):
        """Argument and the value returned are d-dimensional numpy arrays."""
        y = np.copy(x)
        x1, x2 = x[:2]
        y[0], y[1] = x1, x2 + b*x1**2 - 100*b
        return y
    
    def phi_b_inv(y):
        """Argument and the value returned are d-dimensional numpy arrays."""
        x = np.copy(y)
        y1, y2 = y[:2]
        x[0], x[1] = y1, y2 - b*y1**2 + 100*b
        return x
    return phi_b, phi_b_inv

def compose2(f, g):
    return lambda x: f(g(x))


# In[16]:


def apply_to(transformation, pts):
    """Used to generate samples of a twist distribution given samples of a Gaussian one.
    The argument transformation, e.g. phi_b(x1, x2) = (y1, y2) is a 2-dimensional
    transformation of the vectors in pts. The result is an array of the transformed points.
    """
    transformed_pts = np.zeros_like(pts)
    for i, pt in enumerate(pts):
        transformed_pts[i] = transformation(pt)
    return transformed_pts


# In[17]:


def apply(transformation):
    return lambda pts: apply_to(transformation, pts)


# In[18]:


def get_twisted_contour(gaussian, b):
    cov = gaussian['Covariance']
    f = get_Gaussian_contour(cov)
    return lambda x1, x2: f(x1, x2 + b*x1**2 - 100*b)


# In[19]:


def generate_twist(gaussian, b, name):
    # The twisted distribution is a transformation of 
    # the uncorrelated Gaussian distribution 'gaussian'
    transformed_distr = gaussian.copy()
    transformed_function, inverse_twist_function = f_twist(b=b)
    transformed_pdf = compose2(gaussian['pdf'], transformed_function)
    contour_function = get_twisted_contour(gaussian=gaussian, b=b)
    transformed_distr.update({'Name':name, 
                              'Generator':gaussian, 
                              'pdf':transformed_pdf,
                              'Contour Function':contour_function})
    transformed_distr.update({'Transformation':apply(inverse_twist_function)})
    return transformed_distr


# In[20]:


def generate_all_twists(gaussian, b_values, names):
    twists ={name:generate_twist(gaussian, b, name) 
             for b, name in zip(b_values, names)}
    return twists


# In[21]:


def generate_test_suite(sp):
    gaussians = generate_all_Gaussians(sp)
    twists = generate_all_twists(gaussian=gaussians['Pi_1'], 
                                 b_values=[0.03, 0.1], 
                                 names=['Pi_3', 'Pi_4'])
    sp.update({'Test Suite':{**gaussians, **twists}})


# In[22]:


def generate_state_space(dim, nb_runs=100, N=None):
    sp = state_space(dim=dim)
    generate_test_suite(sp)
    generate_initial_states(sp=sp, nb_runs=nb_runs)
    return sp


# ### Generate independent and identically distributed or i.i.d. samples
# 
# These samples will be generated when we initialize a run. They are compared to the correlated samples generated by a MCMC sampler.

# In[23]:


def iid_samples_Gaussian(gaussian, N):
    mean, cov = gaussian['Mean'], gaussian['Covariance']
    rv = ss.multivariate_normal(mean=mean, cov=cov)
    samples = rv.rvs(size=N)
    gaussian.update({'Samples':samples})


# ### Generate i.i.d. samples of an transformed Gaussian distribution.
# These samples will be generated when we initialize a run. They are compared to the correlated samples generated by a MCMC sampler.

# In[24]:


def iid_samples_transformed_Gaussian(distr, N):
    #Samples are generated by transforming the random samples of 
    #the generating Gaussian distribution.
    generator = distr['Generator']
    transformation = distr['Transformation']
    if not 'Samples' in generator:
        iid_samples_Gaussian(generator, N)
    transformed_samples = transformation(generator['Samples'])
    distr.update({'Samples':transformed_samples})


# ## Generate i.i.d. samples for the whole Test Suite

# In[25]:


def generate_iid_samples(sp, N):
    test_suite = sp['Test Suite']
    for name, distr in test_suite.items():
        if 'Generator' not in distr:
            iid_samples_Gaussian(gaussian=distr, N=N)
        else:
            iid_samples_transformed_Gaussian(distr=distr, N=N)


# ## Getter functions for the samples of a distribution

# In[26]:


def get_distribution(sp, name):
    return sp['Test Suite'][name]

def get_samples(sp, name):
    return get_distribution(sp, name)['Samples']


# # Time to test

# In[27]:


def inspect(sp, field):
    test_suite = sp['Test Suite']
    for key, distr in test_suite.items():
        print(key, distr[field])
        
#inspect(SP, 'Covariance')


# In[28]:


def inspect_Gaussian(sp, name_gaussian):
    gaussian = sp['Test Suite'][name_gaussian]
    print(gaussian['Name'])
    print(gaussian['Mean'])
    print(gaussian['Covariance'])
    print(gaussian['Samples'][:5])

def inspect_transformed_Gaussian(sp, name_distr):
    distr = sp['Test Suite'][name_distr]
    print(distr['Name'])
    print(distr['Mean'])
    print(distr['Covariance'])
    inspect_Gaussian(sp, distr['Generator']['Name'])
    print(distr['Samples'][:5])

#inspect_transformed_Gaussian(SP, 'Pi_4')


# SP = generate_state_space(dim=2, nb_runs=10)
# generate_iid_samples(SP, N=1000)
# TESTSUITE = SP['Test Suite']

# Z_samples = get_samples(SP, name='Z')
# 
# prop =  SP['Test Suite']['Proposal']
# prop_cov = prop['Covariance']
# prop_samples = prop['Samples']
# samples = Z_samples @ prop_cov
# 
# samples[:10], prop_samples[:10]
