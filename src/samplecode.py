import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt

import heaviside as hv
from circuits import get_tau, get_rs_lists
from hamiltonians import get_hamiltonian
from binarysearch import acdf, certify_mv, certify_av, invert_cdf
from MonteCarlo import Metropolis

print("test")

H, n_qubits, hf_list, gs = get_hamiltonian('H2', compute_gs=False)

N_sample = 10_000
N_thermalization = 10_000
k_max = 1_001
beta = 1E6
N_heaviside = 161
    
sample = Metropolis(N_thermalization, N_sample, k_max, beta)

r_list, s_list, k_list = get_rs_lists(sample, hamiltonian=H, n_qubits=n_qubits, hf_list=hf_list, tau = get_tau(H))

delta = np.pi/4
eta = 0.4
S = hv.normalization_constant(N_heaviside, beta)

x = invert_cdf(r_list, s_list, k_list, delta, eta, S)
