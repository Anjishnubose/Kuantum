import pennylane as qml
import pennylane.numpy as np

import PauliProducts
import LCUSampling

#full H exponential
def get_U(H, tau, k):
    """
    returns exp(-iH*tau*k)

    """
    hamiltonian = H
    coeff = (-1.j)*tau*k
    return qml.exp(hamiltonian, coeff).matrix()

def prepare_hf_gs(hf_list, start_wire = 0):
    targets = [i for i, a in enumerate(hf_list) if a == 1]
    for wire in targets:
        qml.PauliX(wires=wire + start_wire)

def prepare_initial_state(state_type = 'hf', start_wire=1, **kwargs):
    """
    prepare initial state

    """
    if state_type == 'hf':
        hf_list = kwargs['hf_list']
        prepare_hf_gs(hf_list=hf_list, start_wire=start_wire)
    return

def hadamard_test_randomized(H, n_qubits, n_samples, l_samples, t: float, r: int, term_degree: np.array, measure = 'real', control_wires = [0]):
    """
    Returns Hadamard test samples for randomized Hamiltonian evolution implementation
    
    """

    rotation_pauli, rotation_pauli_signs, pauli_product_red, pauli_product_phase = PauliProducts.reorder_pauli_rotation_products(H, n_samples, l_samples)

    #create vector for angles
    angles = np.array(len(term_degree))
    for i in angles:
        angles[i] = LCUSampling.theta(term_degree[i], t, r)

    #create array to store Pauli rotations
    rotations = np.array(len(rotation_pauli))
    for r in len(rotation_pauli):
        rotations[r] = qml.exp(rotation_pauli, rotation_pauli_signs[r] * angles[r]* 1j)
    
    #hamiltonian_matrix = qml.matrix(H)
    wires = [1 + a for a in range(n_qubits)] #target
    
    qml.Hadamard(wires=control_wires)
    qml.ControlledQubitUnitary(qml.sprod(pauli_product_phase, pauli_product_red), control_wires=control_wires, wires=wires)
    for p in len(rotations):
        qml.ControlledQubitUnitary(rotations[p], control_wires=control_wires, wires=wires)

    #real or imaginary
    if measure=='imag':
        qml.adjoint(qml.S(wires=control_wires))

    qml.Hadamard(wires=control_wires)

def hadamard_test(U, n_qubits, measure='real', control_wires=[0]):
    """
    Returns Hadamard test samples
    
    """
    
    #hamiltonian_matrix = qml.matrix(H)
    wires = [1 + a for a in range(n_qubits)] #target
    
    qml.Hadamard(wires=control_wires)
    qml.ControlledQubitUnitary(U, control_wires=control_wires, wires=wires)

    #real or imaginary
    if measure=='imag':
        qml.adjoint(qml.S(wires=control_wires))
    qml.Hadamard(wires=control_wires)

# import matplotlib.pyplot as plt
def exp_from_samples(samples):
    """
    Returns expectation of Z operator from 0,1 samples
    
    """
    return 1 - 2*np.average(samples)

def get_tau(hamiltonian, norm_bound = np.pi/2):
    """
    Returns suggested tau, given norm bound.
    
    """
    return norm_bound/np.linalg.norm(qml.matrix(hamiltonian))

def make_circuit(U, n_qubits, hf_list, measure='real'):

    control_wires = [0]
    #adds X gates to initialize psi
    prepare_initial_state(state_type='hf', hf_list=hf_list, start_wire=len(control_wires))
    hadamard_test(U, n_qubits=n_qubits, measure=measure, control_wires=control_wires)

    return qml.sample(wires=[0])


def get_gk(k, nk, hamiltonian, n_qubits, hf_list, tau, measure = 'real'):
    """
    Run state prep, hadamard test and get samples and ancilla expectation.

    measure: 'real' / 'imag'

    intializes hf basis ground state
    

    """
    
    dev = qml.device("default.qubit", wires=n_qubits+1, shots=nk)
    circuit_qnode = qml.QNode(make_circuit,dev)

    U = get_U(H=hamiltonian, tau=tau, k=k)

    samples = circuit_qnode(U, n_qubits=n_qubits, hf_list = hf_list, measure=measure)
    #qml.draw_mpl(circuit_qnode)(U, n_qubits=n_qubits, hf_list = hf_list, measure=measure) #to draw circuit
    return 1-2*np.array(samples), exp_from_samples(samples)

def get_randomized_gk(H, k: int, n_qubits, hf_list, tau: float, measure = "real", N_thermalization: int = 10000, reduced_range: int = 10, move_range: int = 20, n_max: int = 200, r: int=1000):
    """
    Run state prep, hadamard test and get samples and ancilla expectation.

    measure: 'real' / 'imag'

    Inputs: H Hamiltonian as linear combination of Paulis, k sample index, n_qubits number of qubits, hf_list Hartree-Fock state, tau time parameter, measure "real" or "imag" (default "real"), N_thermalization reduced_range move_range n_max for LCU sampling, r number of Taylor series terms to sample

    intializes hf basis ground state
    """

    #list of Hamiltonian term coefficients (assuming Hamiltonian is written as linear combination of Paulis)
    pls = H.terms()[0]
    t = -k * tau * np.sum(np.abs(pls))

    n_samples, l_samples = LCUSampling.LCU(r, t, pls, N_thermalization, reduced_range, n_max)
    
    dev = qml.device("default.qubit", wires=n_qubits+1, shots=1)
    circuit_qnode = qml.QNode(make_circuit,dev)

    U = get_randomized_U(H, n_samples, l_samples, t, r)

    samples = circuit_qnode(U, n_qubits=n_qubits, hf_list = hf_list, measure=measure)
    return 1-2*np.array(samples), exp_from_samples(samples)

def get_rs_lists(nk_list, hamiltonian, n_qubits: int, hf_list, tau: float, r: int):
    """
    MAIN FUNCTION, USE THIS!

    returns r_list, s_list, k_list
    
    """

    r_list = np.array([])
    s_list = np.array([])
    index_list = np.array([])

    for i, samples in enumerate(nk_list):
        k = 2*i + 1

        nk = int(samples)
        if nk == 0:
            continue
        if nk == 1: #when shot count is 1, it returns a single number instead of a list smh
            r, expz = get_gk(k, nk, hamiltonian=hamiltonian, n_qubits=n_qubits, hf_list=hf_list, tau=tau, measure='real')
            r_list = np.concatenate((r_list, [r]))

            s, expz = get_gk(k, nk, hamiltonian=hamiltonian, n_qubits=n_qubits, hf_list=hf_list, tau=tau, measure='imag')
            s_list = np.concatenate((s_list, [s]))

            indexes = [k]*nk
            index_list = np.concatenate((index_list, indexes))
        else:
            r, expz = get_gk(k, nk, hamiltonian=hamiltonian, n_qubits=n_qubits, hf_list=hf_list, tau=tau, measure='real')
            r_list = np.concatenate((r_list, r))

            s, expz = get_gk(k, nk, hamiltonian=hamiltonian, n_qubits=n_qubits, hf_list=hf_list, tau=tau, measure='imag')
            s_list = np.concatenate((s_list, s))

            indexes = [k]*nk
            index_list = np.concatenate((index_list, indexes))
    
    return r_list, s_list, index_list