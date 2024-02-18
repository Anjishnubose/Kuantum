import pennylane as qml
import pennylane.numpy as np

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

def get_rs_lists(nk_list, hamiltonian, n_qubits, hf_list, tau):
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
