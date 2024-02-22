import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import yaml as yml

input_file = '/Users/praveenjayakumar/Documents/qhack/Kuantum/inputs/input.yaml'
with open(input_file, 'r') as file:
    inputs = yml.safe_load(file)

print(inputs)

import PhaseEstimation as pe
output  = pe.StatisticalPhaseEstimation(inputs, to_plot=True, plot_path='/Users/praveenjayakumar/Documents/qhack/Kuantum/figures/plot.png')
