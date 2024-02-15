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

    return qml.sample(wires=[0])

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

def get_gk(k, nk, hamiltonian, n_qubits, tau, measure = 'real'):
    """
    Run hadamard test and get samples and ancilla expectation.

    measure: 'real' / 'imag'
    
    """
    
    dev = qml.device("default.qubit", wires=n_qubits+1, shots=nk)
    full_qnode = qml.QNode(hadamard_test,dev)

    U = get_U(H=hamiltonian, tau=tau, k=k)

    samples = full_qnode(U, n_qubits=n_qubits, measure=measure)
    return samples, exp_from_samples(samples)