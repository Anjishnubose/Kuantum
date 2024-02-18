import pennylane as qml
import pennylane.numpy as np

import LCUSampling

def group_paulis(pauli_word_list: np.array, n_samples: np.array, r: int, t: float):
    """
    takes array of indices indicating Pauli words for Pauli rotations, array of angles 
    """

    #create vector for angles
    angles = np.array(len(n_samples))
    for i in angles:
        angles[i] = LCUSampling.theta(n_samples[i], t, r)
    grouping, angle_grouping = qml.pauli.group_observables(pauli_word_list, angles, 'commuting')

    # diagonalizing each grouping

