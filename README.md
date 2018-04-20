# Adaptive Sampling

The project is created to distribute code written as part of my master thesis entitled: ```Adaptive Markov Chain Monte Carlo for Bayesian Inference```.

Adaptive Markov Chain Monte Carlo (MCMC) and stochastic optimization methods are techniques for evaluating intractable integrals. Code for sampling and optimization algorithms, as found in their original papers, has been written in Python and tested against established benchmarks. In addition,  an improvement has been made by incorporating adaptation into stochastic optimization methods, thereby transforming them into samplers. Namely, Gaussian Adaptation (GaA) and Covariance Matrix Adaptation Evolution strategy(CMA-ES) optimizers have been transformed into the Metropolis-GaA, and (1 + 1) CMA, respectively. Performance is quantified using existing convergence and performance measuring tools. Results show adaptive MCMCs with better convergence, mixing, and acceptance ratios.

## Getting Started
You will need Jupyter notebook with Python 3 and the modules listed below. For detailed information and examples of experiment runs, see ```Adaptive_MCMC_for_Bayesian_Inference.pdf```, Chapter 6: Experiments.

### Python modules:

#### Five sampler modules
* Adaptive Metropolis: ``` AM_Sampling.py ```
* Covariance Matrix Adaptation: ```CMA_Sampling.py```
* Gaussian Adaptation: ```GaA_Sampling.py```
* Metropolis Hastings: ```MH_Sampling.py```
* Adaptive MH using Cholesky decomposition of the covariance: ```L_MH_Sampling.py```

#### Test suite module
* Test suites found in Haario et al. (1999): ```TestSuite.py```


#### Supporting modules
* ```FileHandling.py```
* ```Visualization.py```

## Running the tests
The module
```Experiments.ipynb```
offers an easy way to run any of the five samplers and plot their results.

Open the Experiments module in a Jupyter notebook. You will also need the supporting and test modules imported, along with the required libraries as specified in ```Experiments.ipynb```.

### Example
To run the **Adaptive Metropolis** sampler, type ``` AM ``` instead of the current sampler ```CMA```. Sampler names are found in ```Experiments.ipynb``` under **The Samplers**.

In this case, the code:
```
SPEC = specify_run(dim=2, N=10000, name_target='Pi_4', name_sampler='CMA', run_idx=0)
DATA = execute_run(SPEC)
```

becomes:
```
SPEC = specify_run(dim=2, N=10000, name_target='Pi_4', name_sampler='AM', run_idx=0)
DATA = execute_run(SPEC)
```

Also, you might want to change the target distribution. To do so, you need to change ``` name_target```
For reference, see ```Adaptive_MCMC_for_Bayesian_Inference.pdf``` Chapter 6.



## Built With
* [Jupyter](http://jupyter.org/) - Jupyter Notebooks
* [PyMC3](https://docs.pymc.io/) - Bayesian statistical modeling and Probabilistic Machine Learning focusing on advanced MCMC

## Authors

* **Prof. Dr. Bernard Manderik** - *Initial work*


## Acknowledgments

* Hat tip to Nixon Kipkorir Ronoh and Edna Chelangat
Milgo
