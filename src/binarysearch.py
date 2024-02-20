from pennylane import numpy as np
from typing import Callable

def acdf(real_g: np.array, imag_g: np.array, sampled_index: np.array, S: float) -> Callable:
    """
    importance-sampled approximate cumulative distribution function built from samples
    outputs esimator of approximate CDF applied to x
    inputs: real_g, array of Re(Tr(rho U_k)) samples; imag_g, array of Im(Tr(rho U_k)) samples; sampled_index, array of k values of samples; S, normalization constant for importance sampling
    """  
    N_S = len(real_g)

    def H(x: float):
        g = 1/2 + 2 * S * (np.sum(real_g*np.sin(sampled_index*x)) + np.sum(imag_g*np.cos(sampled_index*x))) / N_S #build estimator for ACDF using sampled values from circuits
        return g
    return H

def certify_mv(x: float, real_g: np.array, imag_g: np.array, sampled_index: np.array, eta: float, S: float, N_B: int) -> int:
    """
    certify function by majority vote procedure in LT2022
    checks whether C(x - delta) > eta/2 or C(x - delta) < eta using sampled approximate CDF
    N_B is number of batches in majority vote procedure, M is the total number of samples taken from the circuits
    """
    M = len(real_g)
    cert_output = 0
    #initialize counter in majority vote
    c = 0
    for r in range(N_B):        
        X_r = real_g[r:-1:N_B]
        Y_r = imag_g[r:-1:N_B]
        sampled_index_r = sampled_index[r:-1:N_B]

        g = acdf(X_r, Y_r, sampled_index_r, S)
        if g(x) > 3*eta/4:
            c += 1 #add 1 to vote counter if threshold is met
    #approve if most of majority vote passes 
    if c <= N_B/2: 
        cert_output = 1
    return cert_output

def certify_av(x: float, real_g: np.array, imag_g: np.array, sampled_index: np.array, eta: float, S: float) -> int:
    """
    heuristic certify subroutine only checking average value of ACDF estimate
    certify subroutine, returns 0 if acdf(x) < eta - epsilon, 1 if acdf(x) > epsilon
    """
    g = acdf(real_g, imag_g, sampled_index, S)
    if g(x) < eta/2:
        cert_output = 0
    else:
        cert_output = 1
    return cert_output

def invert_cdf(real_g: np.array, imag_g: np.array, sampled_index: np.array, delta: float, eta: float, 
            S: float, certify_type: str = "mv", 
            x_0: float = -np.pi, x_1: float = np.pi, N_B: int = 10, g: Callable = acdf) -> float: 
    """
    invert CDF using either average certify subroutine or majority vote certify subroutine (default is average)
    """
    while x_1 - x_0 > 2*delta:
        x = (x_0 + x_1)/2 #set x to be midpoint between x_0 and x_1
        #use certify subroutine to check if C(x+(2/3)delta) > 0 or C(x-(2/3)delta) < eta
        if certify_type == "average":
            u = certify_av(x, real_g, imag_g, sampled_index, eta, S) 
        elif certify_type == "mv":
            u = certify_mv(x, real_g, imag_g, sampled_index, eta, S, N_B)
        if u == 0:
            x_1 = x + 2*delta/3 #move new x_1 point closer to x if C(x+(2/3)delta) > 0
        else:
            x_0 = x - 2*delta/3 #move new x_0 point closer to x if C(x-(2/3)delta) < eta
    return (x_0 + x_1)/2


if __name__ == "__main__":
    d = 100
    real_gs = np.random.rand(d+1)
    imag_gs = np.random.rand(d+1)
    sampled_indices = np.array(range(1, 2*d+3, 2) )
    
    acdf_ = acdf(real_gs, imag_gs, sampled_indices, 1.0)
    # print(acdf_(1.0))
    x = certify_av(1.0, real_gs, imag_gs, sampled_indices, 0.1, 1.0)
    print(x)