import scipy as sp
import array as arr
from pennylane import numpy as np
from typing import Callable

#approximate cumulative distribtion function built from samples
#outputs esimator of approximate CDF applied to x
def acdf(X_samples: arr.array, Y_samples: arr.array, theta: arr.array, sampled_j: arr.array, N_S: int, F: float, x: float) -> complex:
    g = 0
    for k in N_S:
        g += (F/N_S) * (X_samples[k] + 1j*Y_samples[k]) * np.exp(1j* (theta[k] + sampled_j[k]*x)) #build estimator for ACDF using sampled values from circuits
    return g

#certify function by majority vote procedure in LT2022
#checks whether C(x - delta) > eta/2 or C(x - delta) < eta using sampled approximate CDF
#N_B is number of batches in majority vote procedure, M is the total number of samples taken from the circuits
def certify_mv(x, X_samples: arr.array, Y_samples: arr.array, theta: arr.array, sampled_j: arr.array, eta: float, F: float, N_B: int, M: int, g: Callable = acdf) -> float:
    cert_output = 0
    #initialize counter in majority vote
    c = 0
    #number of samples per batch
    N_S = np.ceil(M/N_B)
    for r in N_B:
        #create X, Y, theta and j arrays for each batch
        X_r = np.zeros(N_S)
        Y_r = np.zeros(N_S)
        theta_r = np.zeros(N_S)
        sampled_j_r = np.zeros(N_S)
        i = 0
        while r + N_B*i < M:
            X_r[i] = X_samples[r + N_B*i]
            Y_r[i] = Y_samples[r + N_B*i]
            theta_r[i] = theta[r + N_B*i]
            sampled_j_r[i] = sampled_j[r + N_B*i]
            i += 1
        if g(X_r, Y_r, theta_r, sampled_j_r, N_S, F, x) > 3*eta/4:
            c += 1 #add 1 to vote counter if threshold is met
    #approve if most of majority vote passes 
    if c <= N_B/2: 
        cert_output = 1
    return cert_output


#certify subroutine only checking average value of ACDF estimate
#certify subroutine, returns 0 if acdf(x) < eta - epsilon, 1 if acdf(x) > epsilon
def certify_av(x: float, X_samples: arr.array, Y_samples: arr.array, theta: arr.array, sampled_j: arr.array, eta: float, N_S: int, F: float, g: Callable = acdf) -> int:
    if g(X_samples, Y_samples, theta, sampled_j, N_S, F, x) < eta/2:
        cert_output = 0
    else:
        cert_output = 1
    return cert_output

#invert CDF using either average certify subroutine or majority vote certify subroutine, default is average
def invert_cdf(X_samples: arr.array, Y_samples: arr.array, theta: arr.array, sampled_j: arr.array, delta: float, eta: float, M: int, F: float, certify_type: str = "average", x_0: float = -np.pi/3, x_1: float = np.pi/3, N_B: int = 0, g: Callable = acdf) -> float: 
    while x_1 - x_0 > 2*delta:
        x = (x_0 + x_1)/2 #set x to be midpoint between x_0 and x_1
        #use certify subroutine to check if C(x+(2/3)delta) > 0 or C(x-(2/3)delta) < eta
        if certify_type == "average":
            u = certify_av(x, X_samples, Y_samples, theta, sampled_j, eta, M, F, g) 
        elif certify_type == "mv":
            u = certify_mv(x, X_samples, Y_samples, theta, sampled_j, eta, F, N_B, M, g)
        if u == 0:
            x_1 = x + 2*delta/3 #move new x_1 point closer to x if C(x+(2/3)delta) > 0
        else:
            x_0 = x - 2*delta/3 #move new x_0 point closer to x if C(x-(2/3)delta) < eta
    return (x_0 + x_1)/2


    
