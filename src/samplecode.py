import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import yaml as yml


##### Add absolute path to input file.
input_file = 'C:/Users/anjis/Documents/Kuantum/inputs/input.yaml'
with open(input_file, 'r') as file:
    inputs = yml.safe_load(file)

print(inputs)

import PhaseEstimation as pe
output  = pe.StatisticalPhaseEstimation_wtLCU(inputs, 
                                                to_plot=True, plot_path='C:/Users/anjis/Documents/Kuantum/figures/plot_H2_2.png',
                                                to_save=False, save_path='C:/Users/anjis/Documents/Kuantum/outputs/output_H2_LCU.pkl')
