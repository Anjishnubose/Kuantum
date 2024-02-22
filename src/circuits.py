import pennylane as qml
import pennylane.numpy as np
import numpy as num
import time

from pauli import reorder_pauli_rotation_products, get_exp_Pauli, get_Pauli_matrix
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

def hadamard_test_pauli(pauli_product_red, pauli_product_phase):

    return qml.s_prod(pauli_product_phase, pauli_product_red)

def hadamard_test_pauli_matrix(pauli_product_red, pauli_product_phase, n_qubits):
    """
    Return matrix of product of pauli_prod_red and scalar pauli_product_phase
    """
    return get_Pauli_matrix(pauli_product_red, n_qubits=n_qubits) * pauli_product_phase

def hadamard_test_rotations(n_samples, rotation_pauli, rotation_pauli_signs, t: float, r: int):
    """
    
    NEEDS TO BE FIXED (REVERSE ORDER)
    """
    #create vector for angles
    angles = np.zeros(len(n_samples))
    for i in range(len(angles)):
        angles[i] = LCUSampling.theta(n_samples[i], t, r)

    #create array to store Pauli rotations
    rotations = []
    for ir in range(len(rotation_pauli)):
        rotations.append(qml.exp(rotation_pauli[ir], rotation_pauli_signs[ir] * angles[ir]* 1j))

    return qml.prod(*rotations)

def hadamard_test_rotations_matrix(n_samples, rotation_pauli, rotation_pauli_signs, t: float, r: int, n_qubits: int):
    """
    Returns the product of Pauli rotations as Matrix
    
    """
    angles = np.zeros(len(n_samples))
    for i in range(len(angles)):
        angles[i] = LCUSampling.theta(n_samples[i], t, r)
    
    product_rotation = get_Pauli_matrix(qml.Identity(n_qubits), n_qubits)
    for i, P in enumerate(rotation_pauli):
        theta = LCUSampling.theta(n_samples[i], t, r)
        rot = get_exp_Pauli(P, theta = rotation_pauli_signs[i] * theta, return_matrix=True, n_qubits=n_qubits)
        product_rotation = rot @ product_rotation
    return product_rotation

def hadamard_test_randomized(pauli_prod, rotation, measure = 'real'):
    """
    Returns Hadamard test samples for randomized Hamiltonian evolution implementation
    
    """

    control_wires = [0]
    pauli_wires = list(num.array(pauli_prod.wires)+1)
    rotation_wires = list(num.array(rotation.wires)+1)

    qml.Hadamard(wires = control_wires)
    qml.ControlledQubitUnitary(qml.matrix(rotation), wires = rotation_wires, control_wires = control_wires)
    qml.ControlledQubitUnitary(qml.matrix(pauli_prod), wires = pauli_wires, control_wires = control_wires) 
    

    #real or imaginary
    if measure=='imag':
        qml.adjoint(qml.S(wires=control_wires))

    qml.Hadamard(wires=control_wires)

def hadamard_test_randomized_matrix(pauli_prod_matrix, rotation_matrix, n_qubits, measure = 'real'):
    """
    Returns Hadamard test samples for randomized Hamiltonian evolution implementation
    
    """

    control_wires = [0]
    pauli_wires = [i + 1 for i in range(n_qubits)]#list(num.array(pauli_prod.wires)+1)
    rotation_wires = [i + 1 for i in range(n_qubits)]#list(num.array(rotation.wires)+1)

    qml.Hadamard(wires = control_wires)
    qml.ControlledQubitUnitary(rotation_matrix, wires = rotation_wires, control_wires = control_wires)
    qml.ControlledQubitUnitary(pauli_prod_matrix, wires = pauli_wires, control_wires = control_wires) 
    

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

def make_circuit_randomized(pauli_prod, rotations, hf_list, measure = 'real'):
    control_wires = [0]
    #adds X gates to initialize psi
    prepare_initial_state(state_type='hf', hf_list=hf_list, start_wire=len(control_wires))
    hadamard_test_randomized(pauli_prod, rotations, measure)

    return qml.sample(wires=control_wires)

def make_circuit_randomized_matrix(pauli_prod_matrix, rotations_matrix, hf_list, n_qubits, measure = 'real'):
    control_wires = [0]
    #adds X gates to initialize psi
    prepare_initial_state(state_type='hf', hf_list=hf_list, start_wire=len(control_wires))
    hadamard_test_randomized_matrix(pauli_prod_matrix, rotations_matrix, n_qubits=n_qubits, measure= measure)

    return qml.sample(wires=control_wires)


def get_gk(k, nk, hamiltonian, n_qubits, hf_list, tau, measure = 'real'):
    """
    Run state prep, hadamard test and get samples and ancilla expectation.

    measure: 'real' / 'imag'

    intializes hf basis ground state
    

    """
    
    dev = qml.device("lightning.qubit", wires=n_qubits+1, shots=nk)
    circuit_qnode = qml.QNode(make_circuit,dev)

    U = get_U(H=hamiltonian, tau=tau, k=k)

    samples = circuit_qnode(U, n_qubits=n_qubits, hf_list = hf_list, measure=measure)
    #qml.draw_mpl(circuit_qnode)(U, n_qubits=n_qubits, hf_list = hf_list, measure=measure) #to draw circuit
    return 1-2*np.array(samples), exp_from_samples(samples)

def get_randomized_gk(H, k: int, nk:int, n_qubits: int, hf_list, n_samples, l_samples, t, r, measure = "real"):
                    # N_thermalization: int = 10000, reduced_range: int = 10, move_range: int = 20, n_max: int = 200, r: int=1000):
    """
    Run state prep, hadamard test and get samples and ancilla expectation.

    measure: 'real' / 'imag'

    Inputs: H Hamiltonian as linear combination of Paulis, k sample index, n_qubits number of qubits, hf_list Hartree-Fock state, tau time parameter, measure "real" or "imag" (default "real"), N_thermalization reduced_range move_range n_max for LCU sampling, r number of Taylor series terms to sample

    intializes hf basis ground state
    """

    #get Pauli rotations and products
    rotation_pauli, rotation_pauli_signs, pauli_product_red, pauli_product_phase = reorder_pauli_rotation_products(H, n_samples, l_samples)
    rotations = hadamard_test_rotations(n_samples, rotation_pauli, rotation_pauli_signs, t, r)
    pauli = hadamard_test_pauli(pauli_product_red, pauli_product_phase * (-1.0j)**(num.sum(n_samples)))

    dev = qml.device("lightning.qubit", wires=n_qubits+1, shots=nk)
    circuit_qnode = qml.QNode(make_circuit_randomized, dev)

    samples = circuit_qnode(pauli, rotations, hf_list, measure)
    #fig, ax = qml.draw_mpl(circuit_qnode)(pauli, rotations, hf_list, measure) #to draw circuit
    #fig.savefig("C:/Users/sstb2/Kuantum/figures/circuit.png")
    return 1-2*np.array(samples) #, exp_from_samples(samples)

def get_randomized_gk_matrix(H, k: int, nk:int, n_qubits: int, hf_list, n_samples, l_samples, t, r, measure = "real", tau = 1):
                    # N_thermalization: int = 10000, reduced_range: int = 10, move_range: int = 20, n_max: int = 200, r: int=1000):
    """
    Run state prep, hadamard test and get samples and ancilla expectation.

    measure: 'real' / 'imag'

    Inputs: H Hamiltonian as linear combination of Paulis, k sample index, n_qubits number of qubits, hf_list Hartree-Fock state, tau time parameter, measure "real" or "imag" (default "real"), N_thermalization reduced_range move_range n_max for LCU sampling, r number of Taylor series terms to sample

    intializes hf basis ground state
    """

    #get Pauli rotations and products
    start_time = time.time()
    rotation_pauli, rotation_pauli_signs, pauli_product_red, pauli_product_phase = reorder_pauli_rotation_products(H, n_samples, l_samples)
    # print('Time for matrix reorder: {}'.format(time.time() - start_time))
    start_time = time.time()
    rotations_matrix = hadamard_test_rotations_matrix(n_samples, rotation_pauli, rotation_pauli_signs, t, r, n_qubits = n_qubits)
    # print('Time for matrix rotations: {}'.format(time.time() - start_time))
    start_time = time.time()
    pauli_matrix = hadamard_test_pauli_matrix(pauli_product_red, pauli_product_phase * (-1.0j)**(num.sum(n_samples)), n_qubits = n_qubits)
    # print('Time for pauli product: {}'.format(time.time() - start_time))

    #verifying unitary
    U = get_U(H= H, tau = tau, k = k)
    Umatrix = pauli_matrix @ rotations_matrix

    start_time = time.time()
    dev = qml.device("lightning.qubit", wires=n_qubits+1, shots=nk)
    circuit_qnode = qml.QNode(make_circuit_randomized_matrix, dev)

    start_time = time.time()
    samples = circuit_qnode(pauli_matrix, rotations_matrix, hf_list, n_qubits, measure)

    return 1-2*np.array(samples)

def get_rs_lists(nk_list, hamiltonian, n_qubits: int, hf_list, tau: float):
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