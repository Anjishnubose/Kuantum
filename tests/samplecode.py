import pennylane as qml
import pennylane.numpy as np

from Kuantum.src.circuits import get_gk, get_tau, get_rs_lists
from Kuantum.src.hamiltonians import get_hamiltonian

H, n_qubits, hf_list, gs = get_hamiltonian('H2', compute_gs=False)

get_rs_lists([2, 2, 2, 1, 0, 10], hamiltonian=H, n_qubits=n_qubits, hf_list=hf_list, tau = get_tau(H))