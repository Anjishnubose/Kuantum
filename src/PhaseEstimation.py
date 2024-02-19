import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from heaviside import normalization_constant
from circuits import get_randomized_gk
from hamiltonians import read_hamiltonian, decompose
from binarysearch import acdf, invert_cdf
from MonteCarlo import Metropolis
from LCUSampling import LCU


def StatisticalPhaseEstimation(inputs: dict, to_save: bool = False, save_path: str = None, to_plot: bool = False, plot_path: str = None):
    ##### Read the Hamiltonian and the trial state from the destination given in the input file.
    hamiltonian, state = read_hamiltonian(inputs['hamiltonian']['file'])
    ##### decomposing the Hamiltonian as a sum of Pauli words.
    decomposition = decompose(hamiltonian, inputs['hamiltonian']['norm_bound'] * np.pi, inputs['hamiltonian']['error_tolerance'])
    ##### run the importance sampling of the heaviside fourier series.
    samples_k = Metropolis(inputs['sampling k']['num_thermalization'], 
                            inputs['sampling k']['num_sample'],
                            inputs['sampling k']['max_k'],
                            inputs['sampling k']['beta'])
    ##### normalization constant of the heaviside function
    norm_k = normalization_constant(inputs['sampling k']['max_k'], inputs['sampling k']['beta'])
    ##### initializing the shots for the phase estimation
    r_list = np.array([])   ##### real part of the phase
    s_list = np.array([])   ##### imaginary part of the phase
    k_list = np.array([])   ##### k values for the phase estimation
    
    for j, samples in enumerate(samples_k):
        k = 2*j + 1
        num_k = int(samples)
        
        if num_k >0:
            ##### parameters for the LCU sampling depending on the k-value
            t_k = -k*decomposition['tau']*decomposition['lambda']
            r_k = int(np.ceil(2*t_k*t_k))
            ##### LCU sampling to get a list of l-values for each k. 
            ##### Each l-value correspond to a Pauli word in the decomposition of the Hamiltonian.
            samples_n, samples_l = LCU(r_k, t_k, decomposition['coefficients'],
                                        inputs['sampling Unitaries']['num_thermalization'])
            ##### Running the randomly sample LCU decomposition of the time evolution on a quantum circuit.
            r = get_randomized_gk(hamiltonian = hamiltonian,
                                k = k,
                                nk = num_k,
                                n_qubits=decomposition['num_qubits'], 
                                hf_list=state, 
                                samples_l = samples_l,
                                samples_n = samples_n,
                                t = t_k,
                                r = r_k,
                                measure='real')
            s = get_randomized_gk(hamiltonian = hamiltonian,
                                k = k,
                                nk = num_k,
                                n_qubits=decomposition['num_qubits'], 
                                hf_list=state, 
                                samples_l = samples_l,
                                samples_n = samples_n,
                                t = t_k,
                                r = r_k,
                                measure='imag')
            if num_k==1:
                r_list.extend(np.array([r]))
                s_list.extend(np.array([s]))
            else:
                r_list.extend(r)
                s_list.extend(s)
                
            k_list.extend(np.array([k]*num_k))
    ##### Calculating the approximated CDF function
    print("Calculating the approximated CDF function.")
    cdf = acdf(r_list, s_list, k_list, norm_k)
    print("Running the binary search to invert the CDF function.")
    ground_state = invert_cdf(r_list, s_list, k_list, 
                                inputs['binary search']['delta'], inputs['bianry search']['eta'], 
                                norm_k, certify_type='mv') / decomposition['tau']
    print(f"The ground state energy estimate is: {ground_state}.")
    
    if to_plot:
        xs = np.linspace(-inputs['hamiltonian']['norm_bound'] * np.pi, inputs['hamiltonian']['norm_bound'] * np.pi, 501)
        plt.plot(xs, [cdf(x) for x in xs])
        plt.xlabel('x')
        plt.ylabel('CDF(x)')
        plt.savefig(plot_path)
    
    output = dict()
    output['r_list'] = r_list
    output['s_list'] = s_list
    output['k_list'] = k_list
    output['inputs'] = inputs
    output['ground_state'] = ground_state
    output['CDF'] = cdf
    
    if to_save:
        with open(save_path, 'wb') as file:
                pkl.dump(output, file, protocol=pkl.HIGHEST_PROTOCOL)
    
    return output

    
    
    
    
    