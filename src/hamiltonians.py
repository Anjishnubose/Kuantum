#making and importing some hamiltonians

import pennylane as qml
import pennylane.numpy as np


def get_hamiltonian(mol ='h2'):
    if mol == 'h2':
        symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
        H, n_qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            charge=0,
            mult=1,
            basis='sto-3g',
            method='pyscf',
            active_electrons=2,
            active_orbitals=2
        )
        hf_list = [1, 1, 0, 0]
        return H, n_qubits, hf_list