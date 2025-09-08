"""
2 Mode RASER Simulation
Author: Seth Dilday
Editor: Reagan Womack

--- ChatGPT generative AI was used in the development of this script ---
"""
import numpy as np
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the system of 6 coupled ODEs with named variables
def system(t, y):
    d1, d2, a1, a2, phi1, phi2 = y

    T1 = 5.0
    T2 = 2.0
    coupling_beta = 1.0e-22
    deltaNu = 7.1
    nu0 = 1e6
    epsilon = 1e-20

    dd1_dt = -(d1 / T1) - 4 * coupling_beta * (a1**2 + a1 * a2 * np.cos(phi1 - phi2))
    dd2_dt = -(d2 / T1) - 4 * coupling_beta * (a2**2 + a2 * a1 * np.cos(phi1 - phi2))

    da1_dt = -(a1 / T2) + coupling_beta * d1 * (a1 + a2 * np.cos(phi1 - phi2))
    da2_dt = -(a2 / T2) + coupling_beta * d2 * (a2 + a1 * np.cos(phi1 - phi2))

    dphi1_dt = 2 * np.pi * (nu0 + (deltaNu / 2)) + coupling_beta * (d1 / max(a1,epsilon)) * a2 * np.sin(phi2 - phi1)
    dphi2_dt = 2 * np.pi * (nu0 - (deltaNu / 2)) + coupling_beta * (d2 / max(a2,epsilon)) * a1 * np.sin(phi2 - phi1)

    return [dd1_dt, dd2_dt, da1_dt, da2_dt, dphi1_dt, dphi2_dt]

# Time span and initial conditions
t_span = (0, 16)
t_eval = np.linspace(*t_span, 5000)
initial_conditions = [1e20, 1e20, 1e-14, 1e-14, np.pi/2, np.pi/3]  # [d1, d2, a1, a2, phi1, phi2]

print("Starting ODE Solver...")
start = time.perf_counter()
solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, method='BDF')
end = time.perf_counter()
print("ODE Solver done")
print(f"{end - start:.4f} seconds elapsed")

# Extract time and variables
t = solution.t
d1, d2, a1, a2, phi1, phi2 = solution.y

# Construct output signal: coherent field from a1*exp(i*phi1) + a2*exp(i*phi2)
output_signal = (1 / np.sqrt(2)) * (a1 * np.real(np.exp(1j * phi1)) + a2 * np.real(np.exp(1j * phi2)))

# Fourier Transform
freq = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
Y = np.fft.fft(output_signal)
positive_freqs = freq > 0

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Time domain
ax[0].plot(t, output_signal, label='Output Signal')
ax[0].set_title('Time Domain: Output Signal')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)

# Frequency domain
ax[1].plot(freq[positive_freqs], np.abs(Y[positive_freqs]), label='|FFT(Output Signal)|')
ax[1].set_title('Frequency Domain: |FFT(Output Signal)|')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Magnitude')
ax[1].grid(True)

plt.tight_layout()
plt.show()