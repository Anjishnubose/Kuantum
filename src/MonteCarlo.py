import heaviside as hv
from pennylane import numpy as np


"""
function to run a simple Monte Carlo simulation sampling the heaviside function F(k; beta).
args:
    N_thermalization: int - number of steps to thermalize the system.
    N_sample: int - number of samples to take
    N_shot: int - number of shots per sample
    k_max: int - maximum value of k
    beta: float - parameter of the heaviside function
"""
def Metropolis(N_thermalization: int, N_sample: int, k_max: int, 
                beta: float, reduced_range:int = 51, move_range:int = 101) -> np.array:
    """
    This function uses the Metropolis algorithm to sample from the periodic Heaviside function.
    """
        ##### convert range of k in [1, k_max] to range of j in [0, d] where k = 2*j+1, and k_max = 2*d+1.
    d = int((k_max-1)/2)
    j_range = (0, d)
    ##### initialize the sample dict
    sample = np.zeros(d+1, dtype=int)
    probabilities = dict()
    ##### the range to choose an initial value is chosen to be smaller than the full range to avoid vanishing weight.
    reduced_j_range = (0, reduced_range)
    j_current = np.random.randint(*reduced_j_range)
    k_current = 2*j_current+1
    probabilities[int(k_current)] = hv.heaviside_fourier_reduced_mp(k_current, k_max, beta)
    ##### initialize the number of accepted states
    accepted = 0
    ##### run the Metropolis algorithm
    print(f"Running Metropolis algorithm for {N_sample} samples starting at k={k_current}.")
    for step in range(N_thermalization+N_sample):
        # if step%1000 == 0:
        #     print(f"Step {step} of {N_sample}.")
        ##### propose a new state within move_range of the current state.
        j_new = int(j_current + np.random.randint(-min(j_current, move_range), min(d-j_current, move_range)))
        k_new = int(2*j_new+1)
        ##### calculate the acceptance probability
        if k_new not in probabilities:
            probabilities[int(k_new)] = hv.heaviside_fourier_reduced_mp(k_new, k_max, beta)
        
        if probabilities[int(k_current)] == 0:
            p_accept = 1
        else:
            p_accept = probabilities[int(k_new)] / probabilities[int(k_current)]
            p_accept = min(1, p_accept)
        ##### accept or reject the new state randomly
        if np.random.rand() < p_accept:
            ##### change is accepted!
            j_current = np.copy(j_new)
            k_current = np.copy(k_new)
            accepted += 1
        ##### updating the sample count of the new state       
        if step >= N_thermalization:
            sample[int(j_current)] += 1
    ##### returning the samples
    print(f"Sampling ks finished with acceptance rate: {accepted/(N_thermalization+N_sample)}.")
    return sample


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    N_sample = 10_000
    N_thermalization = 10_000
    k_max = 1_001
    beta = 1E6
    
    sample = Metropolis(N_thermalization, N_sample, k_max, beta)
    
    plt.plot(range(1, k_max+2, 2), sample/N_sample, label = r'$\beta = {:.2e}, k_{{max}} = {:d}.$'.format(beta, k_max))
    plt.legend()
    plt.xlabel(r'k')
    plt.ylabel(r'P(k)')
    plt.title("Metropolis Sampling of F(k) with {:.2e} samples.".format(N_sample))
    plt.show()
    # plt.savefig("figures/heaviside_metropolis.png")
    
    
    