#utils and methods with Pauli
import pennylane as qml
import pennylane.numpy as np
import time
#import LCUSampling

def multiply_pauli_list_with_phase(pwds, reverse = False):
    """
    Multiply Pauli words, left to right

    Note, we require the phase since we are performing controlled gates
    """
    if reverse:
        pwds = pwds[::-1]
    
    final_pwd = pwds[0]
    final_phase = 1
    for pw in pwds[1:]:
        final_pwd, phase = qml.pauli.pauli_mult_with_phase(final_pwd, pw)
        final_phase *= phase
    return final_pwd, final_phase

def reorder_pauli_rotation_products(H, n_list, l_list):
    """
    simplify the circuit by pushing the Pauli word products to the end so now the circuit is rotations followed by single product
    
    Parameters:
    l_list: indexes of the Pauli words (rot , non-rot)
    n_list: number of Pauli words in each W_i

    Returns:
    rotation_pauli: list(qml.operator) Pauli word list for exponents
    rotation_pauli_signs: list(+/-1) signs for the exponent angles, including coeff signs
    pauli_product_red: qml.operator Pauli word as a result of product of all Pauli
    pauli_product_phase: extra phase resulting from product, will require to apply an extra Z or S/S^dagger to control wire for -1 or +/-1.j resp, including coeff signs

    """

    #getting Paulis
    pauli_list = []
    pauli_signs = []
    
    hamiltonian_terms = H.terms()[1]
    hamiltonian_coeffs = H.coeffs

    # print(l_list)
    for index in l_list:
        pauli_list.append(hamiltonian_terms[index])
        pauli_signs.append(np.sign(hamiltonian_coeffs[index]))
    
    n_qubits = len(H.wires)
    
    rotation_pauli = []
    rotation_pauli_signs = [] #+/- sign for the pauli exponentials

    pauli_product_red = qml.Identity(wires = n_qubits) #multiplied Pauli word
    pauli_product_phase = 1
    
    # print(l_list)
    # print(n_list)

    for nr in n_list:
        #commute rotation through to start
        rotation_pauli.append(pauli_list[0])
        start_time = time.time()
        if qml.is_commuting(pauli_product_red, pauli_list[0]):
            rotation_pauli_signs.append(1*pauli_signs[0])
        else:
            rotation_pauli_signs.append(-1*pauli_signs[0])
        #print('Time for checking commutation: {}'.format(time.time() - start_time))

        
        #multiply paulis
        start_time = time.time()
        pauli_product_red, phase = multiply_pauli_list_with_phase([pauli_product_red] + pauli_list[1:nr+1], reverse=True)
        #print('Time for pauli product multiplication in reorder: {}'.format(time.time() - start_time))

        pauli_product_phase *= phase
        pauli_product_phase = np.product([pauli_product_phase] + pauli_signs[1:nr+1])

        pauli_list = pauli_list[nr + 1:]
        pauli_signs = pauli_signs[nr + 1:]

    return rotation_pauli, rotation_pauli_signs, pauli_product_red, pauli_product_phase

def get_Pauli_matrix(P, n_qubits):
    wire_map = {i: i for i in range(n_qubits)}
    return qml.pauli.pauli_word_to_matrix(P, wire_map=wire_map)

def get_Pauli_Identity(n_qubits):
    return get_Pauli_matrix(qml.Identity(n_qubits), n_qubits)

def get_exp_Pauli(P, theta, n_qubits, return_matrix = True):
    """
    Exponentiate P as exp(i*theta*P) where P is a Pauli word
    P: qml.Pauli object

    return_matrix:: True for matrix rep over n_qubits
    
    """
    if return_matrix == True:
        I_matrix = get_Pauli_matrix(qml.Identity(n_qubits), n_qubits)
        P_matrix = get_Pauli_matrix(P, n_qubits = n_qubits)
        return np.cos(theta)*I_matrix + 1.j*np.sin(theta)*P_matrix
    else:
        I = qml.Identity(wires = n_qubits)
        return np.cos(theta)*I + 1.j*np.sin(theta)*P

# def full_pauli_rotation(H, rotation_pauli: np.array, rotation_pauli_signs: np.array, term_degree: np,array, t: float, r: int):
#     """
#     Inputs/parameters: Hamiltonian H, rotation_pauli list of Paulis for the Pauli rotations, rotation_pauli_signs list of -1/+1 signs for phases of Pauli rotations, term_degree: array of Taylor series term degree corresponding to each Pauli rotation, t evolution time, r number of Taylor series terms sampled 
#     Outputs operator that is the product of all Pauli rotations with proper phases
#     """
#     #create vector for angles
#     angles = np.array(len(term_degree))
#     for i in angles:
#         angles[i] = LCUSampling.theta(term_degree[i], t, r)
#     n_qubits = len(H.wires)

#     #create array storing Pauli rotations
#     rotations = np.array(len(rotation_pauli))
#     for r in len(rotation_pauli):
#         rotations[r] = qml.exp(rotation_pauli, rotation_pauli_signs[r] * angles[r]* 1j)

#     #multiply all Pauli rotations together
#     multiplied_rotation = qml.prod(*rotations)

#     return multiplied_rotation