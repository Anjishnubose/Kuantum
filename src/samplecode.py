import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import yaml as yml

input_file = 'C:/Users/sstb2/Kuantum/inputs/input.yaml'
with open(input_file, 'r') as file:
    inputs = yml.safe_load(file)

print(inputs)

import PhaseEstimation as pe
output  = pe.StatisticalPhaseEstimation(inputs, to_plot = True, plot_path="C:/Users/sstb2/Kuantum/figures/plot.png")
