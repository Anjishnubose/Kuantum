# Kuantum
QHack 2024

Steps to get the ground state energy estimation (as given in .src/samplecode.py):
1. Install the packages mentioned in the project file.
2. Modify the parameters in the input file given in the inputs folder.
3. Add a path to a pickle file which stores (hamiltonian: qml.Hamiltonian, state: vector) in the input file under 'hamiltonian':file
4. Read the input file from the correct path in samplecode.py
5. [Optional] : Add a plot_path which is a .png file to save a plot of the estimate of the CDF function.
6. [Optional] : Add a save_path which is a .pkl file to save the output which is a dictionary with relevant entries:
    a. 'ground state' which is the estimate of the ground state energy
    b. 'k_list' : the k-values sampled throughout the algorithm
    c. 'r_list' and 's_list': the real and imaginary values of expectation value of the time-evolution unitary taken from the sampled shots from the cosntructed circuits.
    d. 'norm_k' : normalization factor
7. If you want to reconstruct the CDF estimate, just use binarysearch.acdf(r_list, s_list, k_list, norm_k)
   
    
