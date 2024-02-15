#making and importing some hamiltonians

import pennylane as qml
import pennylane.numpy as np

from openfermion import get_ground_state

def get_hamiltonian(mol ='H2', compute_gs = False):
    if mol == 'H2':
        d = 0.74 #ground state separation
        symbols, coordinates = (['H', 'H'], np.array([0., 0., -d/2, 0., 0., d/2]))
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
        hf_list = qml.qchem.hf_state(electrons=2, orbitals = n_qubits)#orbitals here is the spin orbitals
    elif mol == 'LiH':
        d = 1.59 #ground state separation
        symbols, coordinates = (['Li', 'H'], np.array([0., 0., -d/2, 0., 0., d/2]))
        H, n_qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            charge=0,
            mult=1,
            basis='sto-3g',
            method='pyscf',
            active_electrons=4,
            active_orbitals=5
        )
        hf_list = qml.qchem.hf_state(electrons=4, orbitals = n_qubits)
    elif mol == 'H4':
        d = 0.74 #ground state separation
        symbols, coordinates = (['H', 'H', 'H', 'H'], np.array([0., 0., -3*d/2, 0., 0., -d/2, 0., 0., d/2, 0., 0., 3*d/2]))
        H, n_qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            charge=0,
            mult=1,
            basis='sto-3g',
            method='pyscf',
            active_electrons=4,
            active_orbitals=4
        )
        hf_list = qml.qchem.hf_state(electrons=4, orbitals = n_qubits)
    elif mol == 'H2O':
        d = 0.9572
        theta = np.deg2rad(104.52/2)
        symbols, coordinates = (['H', 'O', 'H'], np.array([-np.sin(theta)*d, 0., -np.cos(theta)*d, 0., 0., 0., np.sin(theta)*d, 0., -np.cos(theta)*d]))
        H, n_qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            charge=0,
            mult=1,
            basis='sto-3g',
            method='pyscf',
            active_electrons=4,
            active_orbitals=4
        )
        hf_list = qml.qchem.hf_state(electrons=4, orbitals = n_qubits)

    if compute_gs:
        gs_energy, gs_state = get_ground_state(H.sparse_matrix())
    else:
        gs_energy = None
    
    return H, n_qubits, hf_list, gs_energy