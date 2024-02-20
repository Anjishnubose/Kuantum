#making and importing some hamiltonians

import pennylane as qml
import pennylane.numpy as np
import pickle as pkl
import os
from pathlib import Path

from openfermion import get_ground_state

def hamiltonian_norm(H :qml.qchem.hamiltonian, norm='1'):
    if norm == '1':
        return sum(abs(H.coeff()))
    return

def get_hamiltonian(mol ='H2', compute_gs = False, load = True, save = False, verbose=True):
    if load:
        #loading saved Hamiltonian
        dirname = os.path.dirname(__file__)
        directory = os.path.join(dirname, '../hamiltonians/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        try:
            
            with open(directory + '{}.pkl'.format(mol), 'rb') as file:
                H, hf_list = pkl.load(file)
            if verbose: print('Loaded Hamiltonian at {}{}.pkl, yayy!'.format(directory, mol))
            return H, hf_list
        
        except:
            if verbose: print('Unable to load {}.pkl at {}! Attempting to generate...\n'.format(mol, directory))
    
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
    
    if save:
        #loading saved Hamiltonian
        dirname = os.path.dirname(__file__)
        directory = os.path.join(dirname, '../hamiltonians/')
        #directory = '../hamiltonians/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        try:
            with open(directory + '{}.pkl'.format(mol), 'wb') as file:
                pkl.dump((H, hf_list), file, protocol=pkl.HIGHEST_PROTOCOL)
            if verbose: print('Saved Hamiltonian at {}{}.pkl'.format(directory, mol))
        except:
            raise 'Unable to save hamiltonian'
    
    return H, hf_list

"""
function to read a saved hamiltonian from file.
"""
def read_hamiltonian(file_name: str):
    ##### loading saved Hamiltonian
    try: 
        with open(file_name, 'rb') as file:
            hamiltonian, state = pkl.load(file)
        print('Loaded Hamiltonian at {}, yayy!'.format(file_name))
        return hamiltonian, state
    
    except:
        print('Unable to load {}'.format(file_name))
        return None
    
"""
Decomposes the Hamiltonian into a sum of multi-qubit Pauli matrices.
"""
def decompose(H: qml.Hamiltonian, norm_bound: float = np.pi/2, error_tol: float = 1E-2):

    coefficients, operators = H.terms()
    n_qubits = len(H.wires)
    ##### bound on the norm of the Hamiltonian
    Lambda = np.sum(np.abs(coefficients))
    ##### how much to scale the Hamiltonian by to get its eigenvalues in the range [-norm_bound, norm_bound]
    tau = norm_bound / (Lambda + error_tol)
    return {'coefficients': coefficients, 'operators': operators, 
            'tau': tau, 'lambda': Lambda,
            'num_qubits': n_qubits}
    