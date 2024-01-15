import numpy as np
from numba import jit
from typing import Tuple, Optional
from src.init_parameters import compute_gamma2
import scipy as sp

@jit(nopython=True)
def R2_q_flatted_grid(grid: np.ndarray) -> np.ndarray:
    """
    Flatten a 2D grid of parameters (grid) into a 2D array where each row
    represents a pair of parameters (R2, q).

    Parameters:
        grid (np.ndarray): 1D array representing a 2D grid of parameters.
s
    Returns:
        np.ndarray: 2D array with two columns representing pairs of parameters (R2, q).
    """
    n_pas = len(grid)
    R_q = np.zeros((n_pas**2, 2))
    for i in range(n_pas):
        for j in range(n_pas):
            R_q[i*n_pas+j,0] = grid[i]
            R_q[i*n_pas+j,1] = grid[j]
    return R_q

@jit(nopython=True)
def get_R2_q_grid() -> np.ndarray:
    """
    Generate a 2D grid of parameters (R2, q) with specified discretization.

    Returns:
        np.ndarray: 2D array with two columns representing pairs of parameters (R2, q).
    """
    arr0 = np.arange(0.001,0.101,0.001) # does not start at 0 because division by 0 otherwise
    arr1 = np.arange(0.11,0.91,0.01)
    arr2 = np.arange(0.901,1,0.001)
    discretization =  np.concatenate((arr0, arr1, arr2), axis=0)
    return R2_q_flatted_grid(discretization)

def get_R2_q_densities(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Calculate densities for a 2D parameter grid (R2, q) given data.

    Args:
        data (np.ndarray): Array containing the data.
        grid (np.ndarray): 2D array representing the parameter grid.

    Returns:
        np.ndarray: Array of densities corresponding to the parameter grid.
    """
    k=int(data[0,0])
    R2 = grid[:,0]
    q = grid[:,1]
    s_z = np.sum(data[2,:k]) #z
    
    sigma2_term = 1/(1e-6 + 2*data[0,11]) #sigma2
    vx_term_array = (k*data[0,7]*q*(1-R2))/(1e-8 + R2) #vx
    betat_z_beta = np.dot(data[2,k:2*k].T, np.dot(np.diag(data[2,:k]), data[2,k:2*k]))

    log_weights = - sigma2_term*betat_z_beta*vx_term_array
    log_weights += (s_z+s_z/2+data[0,2]-1)*np.log(q) #a
    log_weights += (k-s_z+data[0,3]-1)*np.log(1-q) #b
    log_weights += (data[0,4]-1-s_z/2)*np.log(R2) #A
    log_weights += (s_z/2+data[0,5]-1)*np.log(1-R2)#B
    
    #stabilisation:
    log_weights -= max(log_weights)
    
    weights = np.exp(log_weights)
    return weights/weights.sum()

def get_q_density(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Calculate density for q given data.

    Args:
        data (np.ndarray): Array containing the data.
        grid (np.ndarray): 2D array representing the parameter grid.

    Returns:
        np.ndarray: Array of densities corresponding to the parameter grid.
    """
    proba_join=get_R2_q_densities(data, grid)
    vect_probas_q=[]
    for j in np.unique(grid[:,1]):  #Loop over q support
        index=np.where(grid[:,1]==j)[0]
        p_q_equal_j = np.sum(proba_join[index, ])
        vect_probas_q.append(p_q_equal_j)
    return np.array(vect_probas_q)

def sample_R2_q_post(grid: np.ndarray, data: np.ndarray, npoints: int, seed: int = None) -> np.ndarray:
    """
    Sample from the posterior distribution of R2 and q using a grid-based approach.

    Args:
        grid (np.ndarray): Grid of R2 and q values.
        data (np.ndarray): Observed data.
        npoints (int): Number of points to sample.
        seed (int, optional): random seed

    Returns:
        np.ndarray: Samples from the posterior distribution.
    """
    if seed is not None:
        np.random.seed(seed=seed)
        
    densities = get_R2_q_densities(data, grid)
    index = np.random.choice(np.arange(grid.shape[0]), npoints, p=densities)
    return grid[index,:]

@jit(nopython=True)
def compute_X_and_W_tilde(z: np.ndarray, data: np.ndarray, return_null_indexes: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute X_tilde_t and W_tilde based on binary variable z and observed data.

    Args:
        z (np.ndarray): Binary variable z.
        data (np.ndarray): Observed data.
        return_null_indexes (bool, optional): Whether to return the indices where z is zero. Default is False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.darray]: Tuple containing X_tilde_t and W_tilde.
    """
    non_zero_z=np.nonzero(z)[0]
    if return_null_indexes:
        zero_z_indexes = np.where(z==0)[0]
    else:
        zero_z_indexes = None
    I_s_z=np.identity(non_zero_z.shape[0])
    X_tilde_t = data[3+non_zero_z,:] 
    W_tilde = X_tilde_t@X_tilde_t.T+(1/data[0,10])*I_s_z #gamma2
    return X_tilde_t, W_tilde, zero_z_indexes
  

@jit(nopython=True)
def yy_minus_betah_w_betah(X_tilde_t: np.ndarray, W_tilde: np.ndarray, data: np.ndarray) -> float:
    """
    Calculate the quantity y.Ty - beta_hat_tilde^T * W_tilde * beta_hat_tilde.

    Args:
        X_tilde_t (np.ndarray): X_tilde transpose matrix.
        W_tilde (np.ndarray): W_tilde matrix.
        data (np.ndarray): Observed data.

    Returns:
        float: The calculated quantity yy - beta^T * W_tilde * beta.
    """
    xy = X_tilde_t @ data[1,:]
    return data[1,:].T @ data[1,:] -(xy.T @ np.linalg.inv(W_tilde) @ xy)


@jit(nopython=True)
def zi_densities(i: int, data: np.ndarray) -> np.ndarray:
    """
    Calculate the densities for binary variable z_i using a specific index.

    Args:
        i (int): Index of the variable z_i.
        data (np.ndarray): Observed data.

    Returns:
        np.ndarray: Normalized densities for z_i.
    """
    k=data[0,0]
    zi = np.arange(2)
    log_weights = zi*(np.log(data[0,9])-np.log(1-data[0,9]))-(zi/2)*np.log(data[0,10])# q gamma2
    z = data[2,:k].copy()
    for zi in range(2):
        z[i] = zi
        X_tilde_t, W_tilde, _ = compute_X_and_W_tilde(z, data, False)
        sign, logabsdet = np.linalg.slogdet(W_tilde)
        log_weights[zi] -= (1/2)*sign*logabsdet
        log_weights[zi] -= (data.shape[1]/2)* np.log( yy_minus_betah_w_betah(X_tilde_t, W_tilde, data) )
    
    #stabilisation
    log_weights -= np.min(log_weights)
    
    weights = np.exp(log_weights)
    
    return weights/weights.sum()

@jit(nopython=True)
def sample_zi(i: int, data: np.ndarray, seed: int = None) -> int:
    """
    Sample the value of zi given the observed data.

    Args:
        i (int): Index for which zi is sampled.
        data (np.ndarray): Observed data.
        seed (int, optional): random seed

    Returns:
        int: The sampled value of zi (0 or 1).
    """
    if seed is not None:
        np.random.seed(seed)
        
    if np.random.rand()<zi_densities(i, data)[0]:
        return 0
    return 1

@jit(nopython=True)
def gibbs_z(data: np.ndarray, n_iter: int, seed=None) -> None:
    """
    Gibbs sampling to update the values of zi in the observed data.

    Args:
        data (np.ndarray): Observed data.
        n_iter (int): Number of Gibbs sampling iterations.
        seed (int, optional): random seed.
    """
    if seed is not None:
        k=int(data[0,0])
        data2 = data.copy()
        for j in range(n_iter):
            for i in range(k):
                data2[2,i] = sample_zi(i, data2, seed*i*j)
    else:
        k=int(data[0,0])
        data2 = data.copy()
        for j in range(n_iter):
            for i in range(k):
                data2[2,i] = sample_zi(i, data2)
    return data2

def sample_sigma2_post(data: np.ndarray, n_variables: int, seed=None) -> np.ndarray:
    """
    Sample from the posterior distribution of sigma2.

    Args:
        data (np.ndarray): Observed data.
        n_variables (int): Number of samples to draw.
        seed (int, optional): random seed

    Returns:
        np.ndarray: Samples from the posterior distribution of sigma2.
    """
    if seed is not None:
        np.random.seed(seed)
        
    k=int(data[0,0])
    T=len(data[1,:])
    X_tilde_t, W_tilde, _ = compute_X_and_W_tilde(data[2,:k], data, False)
    
    inverse_gamma_dist = sp.stats.invgamma(T/2, scale=(1/2)*yy_minus_betah_w_betah(X_tilde_t, W_tilde, data))
    
    return inverse_gamma_dist.rvs(size=n_variables)

def sample_beta_tilde_post(data: np.ndarray, n_variables: int, seed=None) -> np.ndarray:
    """
    Sample from the posterior distribution of beta tilde using the Gibbs sampling.

    Args:
        data (np.ndarray): Observed data.
        n_variables (int): Number of samples to draw.
        seed (int, optional): random seed

    Returns:
        np.ndarray: Samples from the posterior distribution of beta tilde.
    """
    if seed is not None:
        np.random.seed(seed)
        
    k=int(data[0,0])
    X_tilde_t, W_tilde, null_indexes_beta = compute_X_and_W_tilde(data[2,:k], data, True)
    
    return np.random.multivariate_normal(np.linalg.inv(W_tilde)@X_tilde_t@data[1,:], data[0,11]*np.linalg.inv(W_tilde), n_variables), null_indexes_beta

def gibbs_sampler_joint_post(data: np.ndarray, n_iter: int, burn_in_period: int, n_iter_zi: int, debug: bool = False, seed: int = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Gibbs sampler for the joint posterior distribution.

    Args:
        data (np.ndarray): Initial data.
        n_iter (int): Number of iterations for updating R2, q, gamma2, and sigma2.
        burn_in_period (int): number of iterations eliminated from history of samples drawn
        n_iter_zi (int): Number of iterations for updating zi.
        debug (bool, optional): If True, return additional debugging information. Defaults to False.
        seed (int, optional): random seed
        
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Updated data and debug information (if debug=True).
    """
    if seed is not None:
        np.random.seed(seed=seed)
    data1 = data.copy()
    R2_q_grid = get_R2_q_grid()
    k = int(data1[0,0])
    vx = data1[0,7]
    if debug:
        accu = {"R2 post": [], "q post": [], "sigma2 post": [], "beta post": []}
    
    for step in range(n_iter):
        if debug:
            if step>=burn_in_period:
                accu["R2 post"].append(data1[0,8].copy())
                accu["q post"].append(data1[0,9].copy())
                accu["sigma2 post"].append(data1[0,11].copy())
        
        data1[0,8:10] = sample_R2_q_post(R2_q_grid, data1, 1) # update R2, q
        data1[0,10] = compute_gamma2(data1[0,8], data1[0,9], k, vx) # update gamma2
        data1 = gibbs_z(data1, n_iter_zi) #update z
        data1[0,11] = sample_sigma2_post(data1, 1) #update sigma2
        non_zero_z=np.nonzero(data1[2,:k])[0]
        if non_zero_z.shape[0]>0:
            beta_tilde, null_indexes_beta = sample_beta_tilde_post(data1, 1)
            data1[2,non_zero_z] = beta_tilde  #update beta tilde
            if debug:
                if step>=burn_in_period:
                    beta = np.zeros(beta_tilde.shape[1] + null_indexes_beta.shape[0])
                    beta[np.setdiff1d(np.arange(len(beta)), null_indexes_beta)] = beta_tilde
                    beta = beta[np.newaxis, :]
                    accu["beta post"].append(beta.copy())
    if debug:
        return data1, accu
    return data1, None

def get_q_from_gibbs(data: np.ndarray, burn_in_period: int, n_iter: int, n_iter_zi: int, seed: int = None) -> np.ndarray:
    """
    Get q values from multiple runs of the Gibbs sampler.

    Args:
        data (np.ndarray): Initial data.
        burn_in_period (int): number of iterations eliminated from history of samples drawn
        n_iter (int): Number of iterations for updating R2, q, gamma2, and sigma2.
        n_iter_zi (int): Number of iterations for updating zi.
        seed (int, optional): random seed

    Returns:
        np.ndarray: Array of q values obtained from the Gibbs sampler runs.
    """

    data1, accu = gibbs_sampler_joint_post(data, n_iter, burn_in_period, n_iter_zi, debug=True, seed=seed)
    list_posterior_q=accu["q post"]

    return list_posterior_q