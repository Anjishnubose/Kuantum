import pennylane as qml
import pennylane.numpy as np

from ..src.circuits import get_tau, get_rs_lists
from ..src.hamiltonians import get_hamiltonian
from ..src.binarysearch import acdf, certify_mv, certify_av, invert_cdf
from ..src.MonteCarlo import Metropolis
# from ..src import binarysearch as bs

# H, n_qubits, hf_list, gs = get_hamiltonian('H2', compute_gs=False)

# get_rs_lists([2, 2, 2, 1, 0, 10], hamiltonian=H, n_qubits=n_qubits, hf_list=hf_list, tau = get_tau(H))