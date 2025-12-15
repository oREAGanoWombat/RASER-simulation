# === Simulate Single Mode RASER ===
# import functions
import time as timenow
import numpy as np
from datetime import datetime

# record start time
start_time = timenow.time()

# --- Physical Constants and Experimental Parameters ---
mu_0 = 4 * np.pi * 1e-7
h_bar = 1.05457e-34
gamma_h = 2.67522e8
q_factor = 2000
v_s = 0.5 * (1e-2)**3
T1 = 5.0
T2 = 0.7
n_modes = 48
delta_nu = 0.2
Delta = 10.0
nu_0 = 50.0
d0_total = 25e16
sim_duration = 1.0
n_proj = 64
min_freq_hz = 35
max_freq_hz = 65

# coupling beta
cplng_beta_calc = (mu_0 * h_bar * gamma_h**2 * q_factor) / (4 * v_s)
print(f'Calculated Coupling Constant beta: {cplng_beta_calc:.4e}\n')

# --- Run Notes ---
run_notes = (
    f"Sim: {datetime.now().strftime('%Y-%m%d_%H%M%S')} | \n"
    f"T1 = {T1}s, T2 = {T2}s, Q = {q_factor}, Delta = {Delta}Hz, delta nu = {delta_nu}Hz, Number of modes = {n_modes} | \n"
    f"Recon: {n_proj} Proj., {min_freq_hz}-{max_freq_hz}Hz, {sim_duration}s"
)
print(f'Run Notes: {run_notes}\n')