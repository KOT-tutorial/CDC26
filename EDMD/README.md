# EDMD for Autonomous Systems

We train and compare **EDMD with three different dictionaries** and use it for **multi-step prediction** and the computation of **approximate eigenvalues and eigenfunctions**. As an example system, the **uncontrolled Duffing oscillator** is implemented. The code is written to be read alongside Section III.A-III.C of the tutorial paper, and adopts its naming conventions. Notation does currently not perfectly align with the paper for this part of the code.

## Dictionaries

The following dictionaries are implemented in the toolbox:

| abbreviation| dictionary | basis functions |
| --------  | ---------             | ------------                              | 
| rbf       | radial basis function | $\sigma_f^2\exp(-\frac{\|x-c_i\|^2}{2l^2})$, $c_i$ from <tt>kmeans</tt>  | 
| poly      | polynomial features | $\prod_{i\in n} x_i^{d_j}$ |
|fourier    | random Fourier features | $[\sin(\omega_i^\top x), \cos(\omega_i^\top x)]^\top$, $\omega_i\sim\mathcal{N}(0,I)$ 


For the construction of the polynomial dictionary, the function in <tt>poly\_dict.m</tt> is called.

## Folder Structure

The folder contains three scripts and two functions:

```
<tt>run_EDMD_demo.m</tt>            rund EDMD and determine prediction errors
<tt>eval_dictionary_size.m</tt>     rund EDMD for different dictionary sizes
<tt>run_EDMD_eigen_demo.m</tt>      run EDMD, compute approx. eigenfunctions and estimate RoA
<tt>poly_dict.m</tt>                auxiliary function to generate polynomial dictionary
<tt>generate_n_d_grid.m</tt>        auxiliary function to generate grid over state space
```

The two functions (<tt>poly\_dict.m</tt>, <tt>generate\_n\_d\_grid.m</tt>) provide auxiliary functionalities. The scripts (<tt>run_EDMD\_demo.m</tt>, <tt>eval\_dictionary\_size.m</tt>, <tt>run_EDMD\_eigen\_demo.m</tt>) can be run independently and output figures to illustrate the performance of EDMD in the respective scenario. 

Usage with the rbf dictionary requires MATLAB with the Statistics and Machine Learning Toolbox (<tt>kmeans</tt>).

## Quick Start

To use the code, simply run either of the following scripts:
* <tt>run_EDMD\_demo.m</tt> 
* <tt>eval\_dictionary\_size.m</tt>
* <tt>run_EDMD\_eigen\_demo.m</tt> 


## User Settings

Depending on the used script different parameters can be changed. We first introduce the parameters that are shared among the different simulations:

```
%% training data
data_type           choices: trajectory, grid
N_train             number of grid points in each dimension
range_train         lower and upper limits of the initial states for training data
horizon             horizon length for training rollouts; irrelevant for grid data

% dictionary
dictionary_type     choices: poly, fourier, rbf
include_state       determines if the state is explicitly included in dictionary

%% test data
N_test              number of initial states for test data
range_test          lower and upper limits of the test data
hor_pred            length of rollouts for evaluation
reproj_after        number of steps between reprojections
X0                  matrix containing initial states
```


### EDMD Demo

In addition to the previous parameters, the following parameters are relevant in the EDMD demo script.
```
num_feat            number of features (for rbf and fourier dictionaries)
degree              maximum polynomial degree (for poly dictionary)
```

### Ablation on the Dictionary Size

Since we do not fix the number of features in the ablation study on the dictionary size, we have the following parameter instead:

```
dict_range          upper limit for the dictionary size
```

### EDMD Eigenfunction Approximation

When approximating eigenfunctions using EDMD, the adjustable parameters coincide again with the EDMD demo:

```
num_feat            number of features (for rbf and fourier dictionaries)
degree              maximum polynomial degree (for poly dictionary)
```


## Notes

* The $k$-means window selection and random Fourier features make the results seed-dependent; <tt>rng(1)</tt> is fixed for reproducibility of the models themselves.
* The required dictionary size for reproducing the results in the paper differ significantly. Choosing larger numbers typically produce similar results.


## Contact

This section of the KOT Toolbox is maintained by Armin Lederer (National University of Singapore),
armin.lederer@nus.edu.sg. Questions and issues are welcome.
