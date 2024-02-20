import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import yaml as yml

input_file = 'C:/Users/anjis/Documents/Kuantum/inputs/input.yaml'
with open(input_file, 'r') as file:
    inputs = yml.safe_load(file)

print(inputs)

import PhaseEstimation as pe
output  = pe.StatisticalPhaseEstimation_wtLCU(inputs, to_plot=True, plot_path='C:/Users/anjis/Documents/Kuantum/figures/plot.png')
