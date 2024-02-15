from Kuantum.src.circuits import get_gk, get_tau
from Kuantum.src.hamiltonians import get_hamiltonian

import pennylane as qml
import pennylane.numpy as np

H, n_qubits, hf_list = get_hamiltonian('h2')
get_gk(0, 1000, H, hf_list=[1, 1, 0, 0], n_qubits=n_qubits, tau = get_tau(H), measure='imag')