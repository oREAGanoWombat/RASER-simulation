"""
=== SIMULATE RASER ===
Code Authors: Alon Greenbaum, Reagan McNeill Womack
Last Edit: 5 August 2025

Code based on dynamics and theory outlined in DOI: 10.1126/sciadv.abp8483
--- Google Gemini generative AI was used in the development of this script ---
"""

# === REQUIREMENTS ===
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
import os
from datetime import datetime
import time as timenow
import random
from skimage.transform import radon, iradon
from skimage.io import imsave

# === RESULTS DIRECTORY ===
def makedirs(path):
  if not os.path.exists(path):
    os.makedirs(path)

# === CREATE SAMPLE INVERSION MAP ===
def create_sample_inversion_map(shape=(10,10)):
  square_size = (shape[0] // 2, shape[1] // 2) # set size of square using parameter
  inversion_map = np.full(square_size, 1)
  inversion_map = np.pad(inversion_map, ((shape[0] // 4, shape[0] // 4), (shape[1] // 4, shape[1] // 4)), mode="constant", constant_values=0)
  return inversion_map

# === SIMULATE RASER DYNAMICS ===
"""
Core simulation function; models the time evolution of multiple interacting RASER modes based on a set of coupled non-linear ODEs
ODEs are as described in supplementary material at the DOI link at the top of this script
"""
def simulate_raser_dynamics(
        initial_population_inversion=[1e16,1e16],
        T1=5.0,
        T2=0.7,
        coupling_beta=1.0,
        center_freq_hz=50.0,
        gain_bandwidth_hz=10.0,
        mode_spacing_hz=0.2,
        sim_duration=10.0,
        points_per_sec=2000,
):
    N = len(initial_population_inversion)

    # prevent running the simulation on empty projections
    if N == 0 or np.all(initial_population_inversion == 0):
        print("Initial Population Inversion is an empty projection - simulation cancelled")
        return {
            'time': np.linspace(0, sim_duration, int(sim_duration * points_per_sec)),
            'n_modes': N,
            'initial_inversion': initial_population_inversion,
            'final_inversion': np.zeros(N),
            'final_amplitude': np.zeros(N),
            'output_signal': np.zeros(int(sim_duration * points_per_sec)),
        }

    print(f"Running multimode RASER simulation with {N} modes...")
    epsilon = 1e-12 # used to prevent division by zero
    mu_indices = np.arange(1, 2 + 1)
    natural_freq_hz = center_freq_hz - 0.5 * (gain_bandwidth_hz - mode_spacing_hz * (2 * mu_indices - 1))
    omega_natural_rad = 2 * np.pi * natural_freq_hz # first term in Eq. S7

    d0 = initial_population_inversion # initial population inversion for each mode
    A0 = np.random.uniform(0,1e-4,N) # initial amplitudes for each mode, set to tiny random values
    phi0 = np.random.uniform(0,2 * np.pi,N) # initial phases for each mode, set to random values between 0 and 2pi
    y0 = np.concatenate([d0,A0,phi0]) # create 1D array from above 3 arrays in format ODE solver expects

    def raser_ode_system(t, y):
        d = y[0:N]
        A = y[N:2 * N]
        phi = y[2 * N:3 * N]

        X = np.sum(A * np.cos(phi))
        Y = np.sum(A * np.sin(phi))
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        sum_term_1 = (X * cos_phi) + (Y * sin_phi)
        sum_term_2 = (Y * cos_phi) - (X * sin_phi)

        S5 = -(d/T1)-(4*coupling_beta*d*(sum_term_1 ** 2)) # solves Eq. S5, population inversions
        S6 = -(A/T2) + (coupling_beta * d * sum_term_1) # solves Eq. S6, transverse amplitudes
        S7 = omega_natural_rad + (coupling_beta * (d / (A + epsilon))) * sum_term_2 # solves Eq. S7, transverse phases

        dydt = np.concatenate([S5, S6, S7]) # concatenates functions into a 1D array matching structure of y
        return dydt

    t_span = [0, sim_duration] # defines time interval from 0 to sim_duration
    """
    Solve ODEs (ordinary differential equations)
    dense_output = True creates a continuous solution object that can be evaluated at arbitrary time points
    """
    sol = solve_ivp(raser_ode_system, t_span, y0, method='BDF', dense_output=True)

    """
    Process Output Signal
    """
    output_t_points = np.linspace(t_span[0], t_span[1], int(sim_duration * points_per_sec)) # array of time points created using sim_duration and points_per_sec
    output_signal = np.zeros(len(output_t_points))
    final_state_y = sol.sol(output_t_points[-1]) # all d, A, and phi values at last outputTimePoint

    for i, t_point in enumerate(output_t_points): # Signal Calculation Loop
        current_state_y = sol.sol(t_point) # used to retrieve d, A, and phi at current t_point
        A_current = current_state_y[N:2*N] # amplitudes at current_state_y
        phi_current = current_state_y[2*N:3*N] # phases at current_state_y
        output_signal[i] = (1 / np.sqrt(N)) * np.sum(A_current * np.cos(phi_current)) # total RASER signal calculated from Eq. S9

    return {
        'time': output_t_points,
        'n_modes': N,
        'initial_inversion': d0,
        'final_inversion': final_state_y[0:N],
        'final_amplitude': final_state_y[N:2*N],
        'output_signal': output_signal,
    }

# === PLOT RESULTS ===
# creates a 2x2 grid of plots that describe one slice/mode in the simulation
def plot_results(results, png_path):
    time = results['time']
    N = results['n_modes']
    signal = results['output_signal']
    mode_indices = np.arange(1, N + 1) if N > 0 else [] # mu index for plotting

    fig, axes = plt.subplots(2, 2, figsize=(14, 11)) # creates a figure with a 2x2 grid for 4 plots
    fig.suptitle("Multimode RASER Simulation (Single Projection)", fontsize=16)

    # Subplot 1. Population Inversion Profiles (top-left)
    ax1 = axes[0, 0] # select top-left subplot
    if N > 0: # plot two lines if N > 0
        ax1.plot(mode_indices, results['initial_inversion'], 'o-', label=f'$d_\\mu(t=0)$')
        ax1.plot(mode_indices, results['final_inversion'], 'o-', label=f'$d_\\mu(t={time[-1]:.2f}s)$')
    ax1.set_title("Population Inversion Profiles")
    ax1.set_xlabel("Mode Index ($\\mu$)")
    ax1.set_ylabel("Population Inversion ($d_\\mu$)")
    ax1.legend()
    ax1.grid(True)

    # Subplot 2. Final Mode Amplitudes (top-right)
    ax2 = axes[0, 1] # select top-right subplot
    if N > 0:
        ax2.bar(mode_indices, results['final_amplitude'], width=0.8)
    ax2.set_title(f"Final Mode Amplitudes ($A_\\mu$) at t={time[-1]:.2f}s")
    ax2.set_xlabel("Mode Index ($\\mu$)")
    ax2.set_ylabel("Amplitude ($A_\\mu$)")
    ax2.grid(True, axis='y')

    # Subplot 3. Total Output Signal (bottom-left)
    ax3 = axes[1, 0] # select bottom-left subplot
    ax3.plot(time, signal)
    ax3.set_title("Total Output Signal $Sig(t)$ (Eq. S9)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Signal Amplitude (a.u.)")
    ax3.grid(True)

    # Subplot 4. Spectrum of Output Signal (bottom-right)
    min_freq_hz = 35
    max_freq_hz = 65
    ax4 = axes[1, 1] # select bottom-right subplot
    dt = time[1] - time[0]
    signal_fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal_fft), d=dt)
    shifted_freqs = np.fft.fftshift(freqs)
    shifted_magnitudes = np.fft.fftshift(np.abs(signal_fft))
    freq_mask = (shifted_freqs >= min_freq_hz) & (shifted_freqs <= max_freq_hz)
    ax4.plot(shifted_freqs[freq_mask], shifted_magnitudes[freq_mask])
    ax4.set_title("Spectrum of Output Signal")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Magnitude")
    ax4.grid(True)

    # Finalize and save figure
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(png_path)
    plt.show()
    print(f"Saved results as PNG to {png_path}")
    plt.close(fig)

# === PLOT RECONSTRUCTION COMPARISON ===
# creates image of original image and reconstructed image side-by-side
def plot_reconstruction_comparison(original_image, reconstructed_image, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # create figure and 1x2 grid of axes objects
    fig.suptitle("Image Reconstruction Comparison", fontsize=16)

    # Subplot 1. Original Image (left)
    orig = axes[0]
    orig.imshow(original_image, cmap='gray', vmin=0, vmax=1)
    orig.set_title("Original Image")
    orig.set_axis_off()

    # Subplot 2. Reconstructed Image (right)
    recon = axes[1]
    recon.imshow(reconstructed_image, cmap='gray')
    recon.set_title("Reconstructed from RASER Signals")
    recon.set_axis_off()

    # Finalizing and Saving
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path)
    plt.show()
    print(f"Saved reconstruction comparison to {path}")
    plt.close(fig)

# === MAIN EXECUTION BLOCK ===
if __name__ == '__main__':
    # Record start time
    start_time = timenow.time()

    # --- Physical Constants and Experimental Parameters ---
    mu_0 = 4 * np.pi * 1e-7
    h_bar = 1.05457e-34
    gamma_h = 2.67522e8
    T1 = 5.0
    T2 = 0.7
    q_factor = 100
    v_s = 0.5 * (1e-2)**3
    n_modes = 2
    delta_nu = 7.1
    Delta = 10.0
    nu_0 = 50.0
    d0_total = 25e14
    cplng_beta_calc = (mu_0 * h_bar * gamma_h**2 * q_factor) / (4 * v_s)
    print(f'Calculated Coupling Constant beta: {cplng_beta_calc:.4e}\n')

    # Setup Output Directory
    output_directory_root = Path('./Sim_RASER_Output/') # root folder for outputs, using local directory
    model_name = '{}_{}'.format('RASER_reconstruction', datetime.now().strftime("%Y-%m%d_%H%M%S")) # unique name for current run with timestamp
    results_dir = output_directory_root / model_name # full path for current run results
    makedirs(results_dir)
    print(f'Results will be saved at: {results_dir}\n')

    # create log file with important information
    with open(f'{results_dir}/simulation_log.txt', 'w') as log:
        log.write(f'~~~~~~ SIMULATION LOG FOR {results_dir} ~~~~~~\n')
        log.write('\n')
        log.write('--- Physical Constants and Experimental Parameters ---\n')
        log.write(f'mu_0: {mu_0}\n')
        log.write(f'h_bar: {h_bar}\n')
        log.write(f'gamma_h: {gamma_h}\n')
        log.write(f'T1: {T1}\n')
        log.write(f'T2: {T2}\n')
        log.write(f'q_factor: {q_factor}\n')
        log.write(f'v_s: {v_s}\n')
        log.write(f'n_modes: {n_modes}\n')
        log.write(f'delta_nu: {delta_nu}\n')
        log.write(f'Delta: {Delta}\n')
        log.write(f'nu_0: {nu_0}\n')
        log.write(f'd0_total: {d0_total}\n')
        log.write(f'cplng_beta_calc: {cplng_beta_calc}\n')
        log.write('\n')
    print('Simulation log created\n')
    log.close()

    # --- Reconstruction Parameters ---
    n_proj = 60
    min_freq_hz = 35
    max_freq_hz = 65
    print('Reconstruction Parameters Set')
    print(f'Number of Projections: {n_proj}')
    print(f'Frequency Range: {min_freq_hz}-{max_freq_hz}')
    with open(f'{results_dir}/simulation_log.txt', 'a') as log:
        log.write(f'Number of Projections: {n_proj}\n')
        log.write(f'Frequency Range: {min_freq_hz}-{max_freq_hz}\n')
    log.close()

    # Prepare projection for simulation input
    if np.max(projection) > 0:
        projection = projection / np.sum(projection) # normalize
        init_pop_inv = d0_total * projection # convert projection to population inversion
        print('...Projection prepared for simulation')

        # Run RASER simulation
        sim_results = simulate_raser_dynamics(
            initial_population_inversion=init_pop_inv,
            T1=T1,
            T2=T2,
            coupling_beta=cplng_beta_calc,
            center_freq_hz=nu_0,
            gain_bandwidth_hz=Delta,
            mode_spacing_hz=delta_nu,
            sim_duration=2.0,
            points_per_sec=2000
        )
        print(f'RASER simulation complete for current angle')

        # FFT Output Signal
        signal = sim_results['output_signal']
        time = sim_results['time']
        dt = time[1] - time[0]
        signal_fft = np.fft.fft(signal) # perform Fast Fourier Transform (FFT) on output_signal
        freqs = np.fft.fftfreq(len(signal_fft), d=dt) # corresponding frequencies for FFT output
        print('...FFT performed on simulation output')
        shift_freq = np.fft.fftshift(freqs) # shift zero-frequency component to center
        shift_mag = np.fft.fftshift(np.abs(signal_fft)) # magnitude of complex FFT output
        freq_mask = (shift_freq >= min_freq_hz) & (shift_freq <= max_freq_hz) # only include frequencies within range
        filtered_mag = shift_mag[freq_mask] # only include magnitudes in frequency range
        print('1D Projection reconstructed from simulation')

    single_run_results = simulate_raser_dynamics(
        initial_population_inversion=init_pop_inv_rand,
        T1=T1,
        T2=T2,
        coupling_beta=cplng_beta_calc,
        center_freq_hz=nu_0,
        gain_bandwidth_hz=Delta,
        mode_spacing_hz=delta_nu,
        sim_duration=2.0,
        points_per_sec=2000
    )

    # Figure 2. Original vs. Reconstructed Image
    # print("Generating Figure 2. Original vs. Reconstructed Image...")
    # plot_reconstruction_comparison(init_map, recon_img, results_dir / 'Figure2_Reconstruction_Comparison.png')
    # print(f'Figure 2 saved to {results_dir}/Figure2_Reconstruction_Comparison.png\n')

    print('--- SIMULATION COMPLETE ---')
    print('Physical Constants & Experimental Parameters')
    print(f"""
    mu_0: {mu_0}
    h_bar: {h_bar}
    gamma_h: {gamma_h}
    T1: {T1}
    T2: {T2}
    q_factor: {q_factor}
    v_s: {v_s}
    n_modes: {n_modes}
    delta_nu: {delta_nu}
    Delta: {Delta}
    nu_0: {nu_0}
    d0_total: {d0_total}
    cplng_beta_calc: {cplng_beta_calc}
    """)

    # Record end time and calculate time elapsed
    end_time = timenow.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60

    # add time elapsed to simulation_log.txt
    with open(f'{results_dir}/simulation_log.txt', 'a') as log:
        log.write('\n--- SIMULATION DURATION ---\n')
        log.write(f'Simulation Started: {start_time}\n')
        log.write(f'Simulation Ended: {end_time}\n')
        log.write(f'Time Elapsed: {elapsed_seconds:.2f} seconds ({elapsed_minutes:.2f} minutes)')

    print(f'Time Elapsed: {elapsed_seconds:.2f} seconds ({elapsed_minutes:.2f} minutes)')