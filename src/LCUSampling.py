import numpy as np
# import math as mt

"""
normalize a list of Pauli coefficients to a probability distribution.
"""
def normalize_prob(pls: np.array) -> np.array:
    return np.abs(pls) / np.sum(np.abs(pls))

"""
Given a probability distribution, sample n integers from it.
"""
def sample_from_prob(pls: np.array, n: int) -> np.array:
    return np.random.choice(len(pls), n, replace=True, p=np.abs(pls))

"""
probability distribution to sample even integers from.
"""
def qn(n: int, t: float, r:int) -> float:
    return ((t/r)**n)/(n**n * np.exp(-n))*np.sqrt(1+((t/r)/(n+1))**2)

"""
phases of pauli rotations to be used in the LCU sampling.
"""
def theta(n: int, t: float, r: int) -> float:
    return np.arccos(1/(np.sqrt(1+((t/r)/(n+1))**2)))

def LCU_wtMC(r: int, t: float, pls: np.array, qns: np.array,
            ):

    n_samples = np.random.choice(range(2, 2*len(qns)+2, 2), r, replace=True, p=qns) 
    l_samples = np.random.choice(len(pls), np.sum(n_samples)+r, replace=True, p=pls)
    return n_samples, l_samples


"""
Function to sample even integers from the probability distribution q_n. 
For each n, sample n+1 l-values from the given probability distribution and store them.

args:
    r: int - number of samples to take
    t: float - parameter of the probability distribution
    pls: np.array - probability distribution to sample from
    N_thermalization: int - number of steps to thermalize the system.
    reduced_range: int - maximum value of n to initialize from
    move_range: int - range of the move proposal
    n_max: int - maximum value of n to sample from
    
"""
def LCU(r: int, t: float, pls: np.array,
            N_thermalization: int = 0, reduced_range:int = 10, move_range:int = 2, n_max: int = 10
            ):
    
    n_range = np.arange(2, n_max+1, 2)
    reduced_range = np.arange(2, reduced_range+1, 2)
    ##### initializing the random variable n (even integer > 0).
    current_n = np.random.choice(reduced_range)
    ##### getting the probability distribution
    probabilities = normalize_prob(pls)
    ##### empty lists to store the sampled n values
    n_samples = []
    ##### empty list to store the sampled l-values for each n
    l_samples = []
    ##### initialize the number of accepted states
    accepted = 0
    print(f"Running Metropolis algorithm for {r} samples starting at n = {current_n}.")
    for step in range(N_thermalization+r):
        ##### propose a new state within move_range of the current state.
        new_n = int(2*(current_n/2 + np.random.randint(-min(current_n/2-1, move_range), min(n_max/2-current_n/2, move_range))))
        p_accept = qn(new_n, t, r) / qn(current_n, t, r)
        p_accept = min(1, p_accept)
        
        if np.random.rand() < p_accept:
            ##### change is accepted!
            current_n = np.copy(new_n)
            accepted += 1
        ##### Filling the samples once thermalization is done.
        if step >= N_thermalization:
            ##### updating the sample count of the new state
            n_samples.append(int(current_n))
            ##### sampling n+1 l-values from the probability distribution
            ls = sample_from_prob(probabilities, current_n+1)
            l_samples.extend(ls)
    print(f"Sampling ns finished with acceptance rate: {accepted/(N_thermalization+r)}.")
    return n_samples, l_samples
        
        

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    r = 50000
    t = 500
    pls = normalize_prob(np.random.rand(100))
    qns = normalize_prob([qn(n, t, r) for n in range(2, 10, 2)])
    ns, ls = LCU_wtMC(r, t, pls, qns)
    # print(ns)
    # print(ls)
    
    # plt.hist(ns)
    # plt.legend()
    # plt.xlabel(r'n')
    # plt.ylabel(r'f(n)')
    # plt.title("Sampling even integers from q_n")
    # plt.show()