import numpy as np
from numba import jit


def compute_vx(X):
    """Compute mean estimated variance of xt predictors

    Args:
        X (np.array): matrix of xt predictors

    Returns:
        float: mean estimated variance of xt predictors
    """
    return np.mean(np.var(X,axis=0))

def sample_phi(l):
    """Sample phi prior

    Args:
        l (int): number of ut predictors

    Returns:
        int or np.array: phi samples
    """
    if l==0:
        return 0
    else:
        return np.random.uniform(0,1, size=l)
    
def compute_U(T, l):
    """Compute matrix of ut observations

    Args:
        T (int): number of observations
        l (int): number of ut predictors

    Returns:
        int: 0
    """
    if l==0:
        return 0
    else:
        pass

def sample_epsilon(T, sigma2, seed=None):
    """Sample epsilon_1,...,epsilon_T

    Args:
        T (int): number of observations
        sigma2 (float): sigma2 previously sampled
        seed (int, optional): random seed

    Returns:
        np.array: dimensions 1*T
    """
    if seed is not None:
        np.random.seed(seed=seed)
        
    return np.random.normal(loc=0, scale=sigma2, size=T)


def sample_R2(A,B, seed=None):
    """Sample R^2 according to a beta distribution

    Args:
        A (float): shape parameter
        B (float): shape parameter
        seed (int, optional): random seed

    Returns:
        float: R^2 random variable
    """
    if seed is not None:
        np.random.seed(seed=seed)

    return np.random.beta(A,B)

@jit(nopython=True)
def compute_gamma2(R2, q, k, vx):
    """Compute gamma^2 by inverting the R^2 function

    Args:
        R2 (float): R^2 random variable
        q (float): q random variable
        k (int): number of xt predictors
        vx (float): mean estimated variance of xt predictors

    Returns:
        float: gamma^2 random variable
    """
    return R2/((1-R2)*q*k*vx)

def sample_q(a,b, seed=None):
    """Sample q according to a beta distribution

    Args:
        a (float): shape parameter
        b (float): shape parameter
        seed (int, optional): random seed
    Returns:
        float: q random variable
    """
    if seed is not None:
        np.random.seed(seed=seed)

    return np.random.beta(a,b)

def sample_s(k, q):
    return np.random.binomial(k, 1-q)

def sample_beta(q, k, sigma2, gamma2):
    beta=np.zeros(k)
    for i in range(len(beta)):
        bernoulli = np.random.binomial(1, q)
        if bernoulli == 1:
            beta[i] = np.random.normal(0, np.sqrt(sigma2*gamma2))
    return beta

def compute_Z(beta):
    """Compute z_1,...,z_k

    Args:
        beta (np.array): random vector beta

    Returns:
        np.array: dimensions1*k
    """
    Z=beta
    Z[Z!=0]=1
    return Z

def compute_Y(X, beta, epsilon):
    """Compute y_1,...,y_T

    Args:
        X (np.array): matrix of xt predictors
        beta (np.array): random vector beta
        epsilon (np.array): vector of epsilon_1,...,epsilon_T

    Returns:
        np.array: dimensions 1*T
    """
    return X@beta + epsilon


### Final function
def init_parameters(T, k, l, a, b, A, B, X, sigma2, seed=None):
    """
    Initialize parameters for a given simulation.

    Args:
        seed (int): Seed for reproducibility.
        T (int): Number of observations.
        k (int): Number of covariates.
        l (int): Number of latent variables.
        a (float): Shape parameter for gamma2.
        b (float): Shape parameter for gamma2.
        A (float): Shape parameter for q.
        B (float): Shape parameter for q.
        seed (int, optional): random seed.

    Returns:
        dict: Dictionary containing initialized parameters.
    """
    q = sample_q(a,b)
    R2 = sample_R2(A,B, seed=seed)
    vx=compute_vx(X)
    gamma2 = compute_gamma2(R2=R2, q=q, k=k, vx=vx)

    dct = {
        "X" : X,
        "U": compute_U(T=T, l=l),
        "beta": sample_beta(q=q, k=k, sigma2=sigma2, gamma2=gamma2),
        "phi": sample_phi(l=l),
        "q": q
    }
    dct["R2"] = R2
    dct["gamma2"] = gamma2
    dct["Z"]=compute_Z(beta=dct["beta"])
    dct["sigma2"] = sigma2
    dct["epsilon"] = sample_epsilon(T=T, sigma2=dct["sigma2"], seed=seed)
    dct["Y"]=compute_Y(X=dct["X"], beta=dct["beta"], epsilon=dct["epsilon"])
    return dct