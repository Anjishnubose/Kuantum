import mpmath as mp
from pennylane import numpy as np
import scipy as sp


"""
##### Defining the fourier transform of the periodic Heaviside function.
##### this dispatch uses scipy's bessel functions which are faster than mpmath's.
##### the scipy bessel functions however return 
"""    
def heaviside_fourier(k: int, N: int, beta: float) -> np.complex128:

    if k == 0:
        return 0.5
    ##### when k = 1, 3, ...,N-2 odd integers
    elif k < N and k%2 == 1:
        j = int((k-1)/2)
        return -1.0j * ((sp.special.iv(j, beta) + sp.special.iv(j+1, beta))/(k) * np.exp(-beta) * np.sqrt(beta / (2*np.pi)))
    ##### when k = N
    elif k == N and N%2 == 1:
        j = N
        return -1.0j * ((sp.special.iv(j, beta))/(k) * np.exp(-beta) * np.sqrt(beta / (2*np.pi)))
    else:
        return 0.0


"""
This dispatch is needed since mpmath's bessel functions are more accurate than scipy's.
"""
def heaviside_fourier_reduced_mp(k: int, N: int, beta: float) -> np.float64:

    ##### when k = 1, 3, ...,N-2 odd integers
    if k < N and k%2 == 1:
        j = int((k-1)/2)
        return (mp.besseli(j, beta) + mp.besseli(j+1, beta))/(k)
    ##### when k = N
    elif k == N and N%2 == 1:
        j = int((N-1)/2)
        return (mp.besseli(j, beta))/(k)
    else:
        return 0.0

"""
function to calculate the normalization constant for the periodic Heaviside function in fourier space.
"""
def normalization_constant(N: int, beta: float) -> float:
    ##### Since we are dealing with extremely small or large numbers, we use a clever math trick to calculate the sum.
    ##### We want S(beta, N) = sum_{k=1}^{N} |F_k(beta)|.
    ##### However, each F_k(beta) has a factor which goes as exp(-beta).
    ##### For large beta, this number underflows and hence F_k just shows up as 0.
    ##### Instead we write the sum as, S(beta, N) = exp(log(|F_1(beta)|)) * sum_{k=1}^{N} |F_k(beta)|/|F_1(beta)|.
    ##### All the ratios of |F_k/F_1| are now well defined number within computational range and hence that sum is doable.
    ##### The log of |F_1(beta)| is taken analytically to avoid underflow. This number is O(1), and hence its exponential can be taken.
    lf1 = -beta + 0.5*mp.log(beta/(2*np.pi)) + mp.log(mp.besseli(0, beta)+mp.besseli(1, beta))
    
    ratio_sum = 0.0
    for k in range(1, N+2, 2):
        ratio_sum += heaviside_fourier_reduced_mp(k, N, beta)/heaviside_fourier_reduced_mp(1, N, beta)
    
    return float(mp.exp(lf1) * ratio_sum)

"""
Defining the periodic Heaviside function."
"""
def heaviside(x: float, N:int, beta: float) -> np.float64:
    return np.real(np.sum([heaviside_fourier(k, N, beta) * np.exp(1.0j*k*x) for k in range(-N, N+1, 2)]))

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    Ns = [41, 81, 161]
    betas = [2.0, 4.0, 8.0]
    xs = np.linspace(-0.5*np.pi, 0.5*np.pi, 101)
    
    for N in Ns:
        for beta in betas:
            print("N = {:d}, beta = {:f}".format(N, beta))
            ys = [heaviside(x, N, beta) for x in xs]
            plt.plot(xs, ys, label = r"N = {:d}, beta = {:f}".format(N, beta))

    plt.xlabel(r"x")
    plt.ylabel(r"F(x)")
    plt.title("Periodic Heaviside function Estimation")
    plt.legend()
    plt.vlines([0.0], -0.5, 0.5, linestyles = "dashed")
    plt.savefig("figures/heaviside_test.png")
    
