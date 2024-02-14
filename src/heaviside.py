import mpmath as mp
from pennylane import numpy as np
import scipy as sp


"""
Defining the fourier transform of the periodic Heaviside function.
"""
def heaviside_fourier(k: int, N: int, beta: float) -> np.complex128:

    if k == 0:
        return 0.5
    ##### when k = 1, 3, ...,N-2 odd integers
    elif k < N and k%2 == 1:
        j = int((k-1)/2)
        return -1.0j * ((mp.besseli(j, beta) + mp.besseli(j+1, beta))/(k) * np.exp(-beta) * np.sqrt(beta / (2*np.pi)))
    ##### when k = N
    elif k == N and N%2 == 1:
        j = N
        return -1.0j * ((mp.besseli(j, beta))/(k) * np.exp(-beta) * np.sqrt(beta / (2*np.pi)))
    else:
        return 0.0
    
##### this dispatch uses scipy's bessel functions which are faster than mpmath's.
##### the scipy bessel functions however return 
def heaviside_fourier_scipy(k: int, N: int, beta: float) -> np.complex128:

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
The relevant part of the fourier transform of the periodic Heaviside function useful for importance sampling.
"""
def heaviside_fourier_reduced(k: int, N: int, beta: float) -> np.float64:

    ##### when k = 1, 3, ...,N-2 odd integers
    if k < N and k%2 == 1:
        j = int((k-1)/2)
        return (mp.besseli(j, beta) + mp.besseli(j+1, beta))/(k)
    ##### when k = N
    elif k == N and N%2 == 1:
        j = N
        return (mp.besseli(j, beta))/(k)
    else:
        return 0.0

def heaviside_fourier_reduced_scipy(k: int, N: int, beta: float) -> np.float64:

    ##### when k = 1, 3, ...,N-2 odd integers
    if k < N and k%2 == 1:
        j = int((k-1)/2)
        return (sp.special.iv(j, beta) + sp.special.iv(j+1, beta))/(k)
    ##### when k = N
    elif k == N and N%2 == 1:
        j = N
        return (sp.special.iv(j, beta))/(k)
    else:
        return 0.0

"""
function to calculate the ratio of the fourier coefficients of the periodic Heaviside function.
"""
def heaviside_fourier_ratio(k1: int, k2: int, N: int, beta: float) -> np.float64:

    assert k1>0 and k2>0, "k1 and k2 must be positive integers."
    j1 = int((k1-1)/2)
    j2 = int((k2-1)/2)
    
    if k1 != N and k2 != N:
        ratio = ((mp.besseli(j2, beta) + mp.besseli(j2+1, beta))/(k2)) / ((mp.besseli(j1, beta) + mp.besseli(j1+1, beta))/(k1))
    elif k1 == N and k2 != N:
        ratio = ((mp.besseli(j2, beta) + mp.besseli(j2+1, beta))/(k2)) / ((mp.besseli(j1, beta))/(k1))
    elif k1 != N and k2 == N:
        ratio = ((mp.besseli(j2, beta))/(k2)) / ((mp.besseli(j1, beta) + mp.besseli(j1+1, beta))/(k1))
    else:
        ratio = 1.0
    
    return np.abs(ratio)

"""
Defining the periodic Heaviside function."
"""
def heaviside(x: float, N:int, beta: float) -> np.float64:
    return np.real(np.sum([heaviside_fourier_scipy(k, N, beta) * np.exp(1.0j*k*x) for k in range(-N, N+1, 2)]))

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    # Ns = [41, 81, 161]
    # betas = [2.0, 4.0, 8.0]
    # xs = np.linspace(-0.5*np.pi, 0.5*np.pi, 101)
    
    # for N in Ns:
    #     for beta in betas:
    #         print("N = {:d}, beta = {:f}".format(N, beta))
    #         ys = [heaviside(x, N, beta) for x in xs]
    #         plt.plot(xs, ys, label = r"N = {:d}, beta = {:f}".format(N, beta))

    # plt.xlabel(r"x")
    # plt.ylabel(r"F(x)")
    # plt.title("Periodic Heaviside function Estimation")
    # plt.legend()
    # plt.vlines([0.0], -0.5, 0.5, linestyles = "dashed")
    # plt.savefig("figures/heaviside_test.png")
    k_max = 1_001
    beta = 1E2
    # xs = np.array(range(1, k_max+2, 2))
    # print(xs)
    # ys = [np.abs(heaviside_fourier_reduced_scipy(k, k_max, beta)) for k in xs]
    # print(ys)
    # plt.plot(xs, ys, label = r'$\beta = {:.2e}, k_{{max}} = {:d}.$'.format(beta, k_max))
    # plt.show()
    k1 = 5
    k2 = 9
    print(heaviside_fourier_reduced(k2, k_max, beta)/heaviside_fourier_reduced(k1, k_max, beta))
    print(heaviside_fourier_reduced_scipy(k2, k_max, beta)/heaviside_fourier_reduced_scipy(k1, k_max, beta))
    
