
# coding: utf-8

# In[1]:


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


# # *Visualize* the results of an *MCMC* run
# 
# ## Set Up the Grid
# 
# The values of $x_{min}, x_{max}, {nb}_{x}, y_{min}, y_{max}, \text{ and } {nb}_{y}$ depend on where the **probability mass** of the **target distribution** is located, i.e. where the probability density function is sufficiently 'large'.

# In[2]:


Grid = namedtuple('Grid', ['x_min', 'x_max', 'y_min', 'y_max', 'X', 'Y'])

def make_grid(x_min=-30.0, x_max=30.0, nb_x =100, 
              y_min=-30.0, y_max=30.0, nb_y =100):
    x_list = np.linspace(x_min, x_max, nb_x)
    y_list = np.linspace(y_min, y_max, nb_y)
    x, y = np.meshgrid(x_list, y_list)
    return Grid(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, X=x, Y=y)

GRID = make_grid()


# In[4]:


def Mahalanobis_distance(mean, point, precision):
    # The precision matrix is the inverse of the covariance matrix.
    delta = mean - point
    return np.sqrt(delta @ precision @ delta.T)

def squared_Mahalanobis_distance(point, precision):
    # The precision matrix is the inverse of the covariance matrix.
    delta = mean - point
    return delta @ precision @ delta.T

def Mahalanobis_distance_to_origin(point, precision):
    # The precision matrix is the inverse of the covariance matrix.
    return np.sqrt(point @ precision @ point.T)

def squared_Mahalanobis_distance_to_origin(point, precision):
    # The precision matrix is the inverse of the covariance matrix.
    return point @ precision @ point.T

def calculate_fractions(distribution, samples, burnin_pct=0):
    precision = la.inv(distribution['Covariance'])
    end_burnin = burnin_pct*len(samples)//100
    samples_at_equilibrium = samples[end_burnin:]
    nb_samples = len(samples_at_equilibrium)
    d_sq = [squared_Mahalanobis_distance_to_origin(sample, precision) 
            for sample in samples_at_equilibrium]
    return [sum(d_sq <= contour_level)/nb_samples 
            for contour_level in distribution['Contour Levels']]


# # the histogram of the distances
# n, bins, patches = plt.hist(Distances, 50, normed=1, facecolor='green', alpha=0.75)
# 
# plt.xlabel('Distance to the Mean')
# plt.ylabel('Relative Frequency')
# plt.title(r'$\mathrm{Histogram\ of\ Sample\ Distance\ to\ the\ Mean}$')
# plt.grid(True);

# # cumulative distribution of the distances
# values, base = np.histogram(Distances, bins=100)
# # evaluate the cumulative
# cumulative = np.cumsum(values)
# # plot the cumulative function
# plt.plot(base[:-1], cumulative, c='blue');

# fig = plt.figure("i.i.d.", figsize=(7, 7))
# ax = fig.add_subplot(1, 1, 1) 
# subplot(ax, Pi_2, Pi_2.Samples[::1000], dim1=0, dim2=1, 
#         title='Distribution of i.i.d. generated samples.')

# ## Contour Lines corresponding with given Confidence Levels
# 
# Next we plot the contour lines corresponding with 10, 90, 95 and 99 percent confidence. Therefore we
# use the corresponding values of $\chi^2$-distribution. In case of a bivariate distribution we have 2 **degrees of freedom**. The values of this distribution can be found at the webpage https://people.richland.edu/james/lecture/m170/tbl-chi.html for instance.

# In[23]:

def plot_contour_lines(ax, distribution, dim1, dim2):
    global GRID
    X, Y = GRID.X, GRID.Y
    # Plot the contour lines
    contour_function = distribution['Contour Function']
    # Since we project and a 2-dimensional subspace we will use 2 degrees of freedom
    # instead of the dimension of the statespace as we did before.
    contour_levels = distribution['Contour Levels']
    #chi_squares = distribution.ChiSquares
    Z = contour_function(X, Y)
    ax.contour(X, Y, Z, contour_levels)
    
def scatter_samples(ax, samples, dim1, dim2):
    ax.scatter(samples[:, dim1], samples[:, dim2])
    
def subplot(ax, distribution, samples, dim1, dim2, title, fraction_str=None):
    ax.set_title(title, fontweight='bold', color='blue', fontsize=14)
    ax.axis([GRID.x_min, GRID.x_max, GRID.y_min, GRID.y_max])
    ax.set_xlabel('Dimension ' + str(dim1))
    ax.set_ylabel('Dimension ' + str(dim2))
    plot_contour_lines(ax, distribution, dim1, dim2)
    scatter_samples(ax, samples, dim1, dim2)


# In[7]:


def compare_to_iid_samples(run_data, nb_samples, dim1=0, dim2=1, burnin_pct=50):
    global GRID    
    fig, ((ax_left, ax_right)) = plt.subplots(nrows=1, ncols=2, figsize=(15,7))
    target = run_data.Target
    # Data to be plotted.
    step = run_data.N//nb_samples
    mcmc_samples = run_data.DataStore['States']
    iid_samples = target['Samples']
    mcmc_samples_2_display = mcmc_samples[::step]
    iid_samples_2_display = iid_samples[::step]
    mcmc_fractions = calculate_fractions(target, mcmc_samples, burnin_pct)
    iid_fractions = calculate_fractions(target, iid_samples, burnin_pct)
    
    # Information to be shown.
    s_name = run_data.Sampler['Name']
    title_str = 'Distribution of samples generated by {:s}'
    title_info = title_str.format(s_name)
    burnin_str = 'Burn in used is {:d} percent of the generated samples.'
    burnin_info = burnin_str.format(burnin_pct)
    mcmc_str = '{:s} Fractions: {:1.5f}, {:1.5f}, {:1.5f}, and {:1.5f}'
    mcmc_info = mcmc_str.format(s_name, *mcmc_fractions) 
    iid_str = 'IID Fractions: {:1.5f}, {:1.5f}, {:1.5f}, and {:1.5f}'
    iid_info = iid_str.format(*iid_fractions)
    title_mcmc = '{:s} Generated'.format(s_name)
    title_idd = 'IID Generated'
    suptitle_str = 'Comparison of the {:s} (left) vs. the IID (right) sample distribution'
    suptitle = suptitle_str.format(s_name)
    
    # Display everything.
    print(burnin_info)
    print(mcmc_info)
    print(iid_info)
    fig.suptitle(suptitle, fontweight='bold', color='red', fontsize=18)
    subplot(ax_left, target, mcmc_samples_2_display, dim1, dim2, title=title_mcmc)
    subplot(ax_right, target, iid_samples_2_display, dim1, dim2, title=title_idd)


# In[6]:


def plot_samples(run_data, nb_samples, dim1=0, dim2=1, burnin_pct=50):
    global GRID
    # New figure window for the current sampling method
    s_name = run_data.Sampler['Name']
    fig = plt.figure(s_name, figsize=(7, 7)) 
    ax = fig.add_subplot(1, 1, 1)
    # Data to be plotted.
    target = run_data.Target
    # Data to be plotted.
    step = run_data.N//nb_samples
    mcmc_samples_2_display = run_data.DataStore['States'][::step]
    # Information to be shown.
    fig_title_str = 'Distribution of samples generated by {:s}'
    fig_title =  fig_title_str.format(s_name)
    #Plot everything.
    subplot(ax, target, mcmc_samples_2_display, dim1, dim2, title=fig_title)
    
def subplot_2(ax, samples, dim1, dim2, title, color):
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel('Dimension ' + str(dim1))
    ax.set_ylabel('Dimension ' + str(dim2))
    ax.scatter(samples[:, dim1], samples[:, dim2], color=color)
    
def compare_sample_spread(dim1, dim2, list_of_samples, titles, colors):
    # Ensure that dim1 and dim2 are less than the dimension of the state space.
    _, dim = list_of_samples[0].shape
    assert dim1 < dim, "dim1 should be less then %r" % dim
    assert dim2 < dim, "dim2 should be less then %r" % dim
    
    #Generate the supplots.
    fig, (axes) = plt.subplots(nrows=1, ncols=2, figsize=(15,7), sharex='col', sharey='row')
    for ax, samples, title, color in zip(axes, list_of_samples, titles, colors):
        subplot_2(ax=ax, samples=samples, dim1=dim1, dim2=dim2, title=title, color=color)

