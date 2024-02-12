# Kuantum
QHack 2024

Algorithm Overview

Inputs:
- Hamiltonian H, state psi (in PennyLane's format)
- N (maximum value of k), beta (parameter in Heaviside function approximation)
- N_S (number of k samples), N_shot (number of shots of Hadamard test for each sampled k), r (number of Trotter steps)

for i in 1:N_S
  randomly pick k_new in [-N, N]
  calculate |F_(k_new)|/|F_(k_old)|
  if change is accepted
    update k to k_new
  store k in klist

for j in klist
  for l in 1:N_shot
    run Hadamard test circuit
  compute g_k

compute approximate CDF
    
