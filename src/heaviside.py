import scipy as sp
from pennylane import numpy as np


"""
Defining the fourier transform of the periodic Heaviside function.
"""
def heaviside_fourier(k: int, N: int, beta: float) -> complex:

    if k == 0:
        return 0.5
    ##### when k = 1, 3, ...,N-2 odd integers
    elif k < N and k%2 == 1:
        j = (k-1)/2
        return -1.0j * (sp.special.iv(j, beta) + sp.special.iv(j+1, beta))/(k) * np.exp(-beta) * np.sqrt(beta / (2*np.pi))
    ##### when k = N
    elif k == N and N%2 == 1:
        j = N
        return -1.0j * (sp.special.iv(j, beta))/(k) * np.exp(-beta) * np.sqrt(beta / (2*np.pi))
    else:
        return 0.0
    

"""
Defining the periodic Heaviside function."
"""
def heaviside(x: float, N:int, beta: float) -> float:
    return np.real(np.sum([heaviside_fourier(k, N, beta) * np.exp(1.0j*k*x) for k in range(-N, N+1, 2)]))

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    Ns = [41, 81, 161]
    betas = [2.0, 4.0, 8.0]
    xs = np.linspace(-0.5*np.pi, 0.5*np.pi, 101)
    
    for N in Ns:
        for beta in betas:

            ys = [heaviside(x, N, beta) for x in xs]
            plt.plot(xs, ys, label = f"N = {N}, beta = {beta}")

    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Periodic Heaviside function Estimation")
    plt.legend()
    plt.vlines([0.0], -0.5, 0.5, linestyles = "dashed")
    plt.savefig("figures/heaviside_test.png")