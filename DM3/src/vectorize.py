import numpy as np
from typing import Dict
from src.init_parameters import compute_vx

def vectorize_data(dct: Dict[str, np.ndarray], 
                   T: int, 
                   k: int, 
                   l: float, 
                   a: float, 
                   b: float,
                   A: float, 
                   B: float,
                   seed: int = None) -> np.ndarray:
    """
    Vectorize data from a dictionary and additional parameters.

    Parameters:
        dct (Dict[str, np.ndarray]): Dictionary containing data arrays.
        T (int): Number of time points.
        k (int): Number of components.
        l (float): Some parameter.
        a (float): Some parameter.
        b (float): Some parameter.
        A (float): Some parameter.
        B (float): Some parameter.
        seed (int, optional): random seed.

    Returns:
        np.ndarray: Vectorized data array.
    """
    
    data = np.zeros((k+3, T))
    
    data[0,0]=k
    data[0,1]=l
    data[0,2]=a
    data[0,3]=b
    data[0,4]=A
    data[0,5]=B
    data[0,6]=seed
    
    data[0,7]=compute_vx(dct["X"])
    data[0,8]=dct["R2"]
    data[0,9]=dct["q"]
    data[0,10]=dct["gamma2"]
    data[0,11]=dct["sigma2"]
    
    data[1,:]=dct["Y"]
    data[2,:k] = dct["Z"]
    data[2,k:2*k] = dct["beta"]
    data[3:,:] = dct["X"].T
    
    return data