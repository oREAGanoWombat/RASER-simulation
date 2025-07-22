
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import rotate
from pathlib import Path
import os
from datetime import datetime
import random
from skimage.transform import radon, iradon
from skimage.io import imsave



# this function makes the directory to save the results
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_sample_inversion_map(shape=(10, 10)):
    """
    Creates a sample 2D inversion map with a square shape for demonstration.
    """
    square_size = (shape[0] // 2, shape[1] // 2)
    inversion_map = np.full(square_size, 1.0)
    inversion_map = np.pad(inversion_map, ((shape[0] // 4, shape[0] // 4), (shape[1] // 4, shape[1] // 4)),
                           mode="constant", constant_values=0)
    return inversion_map

def simulate_laser_dynamics(
        initial_population_inversion,
        T1=5.0,
        T2=0.7,
        coupling_beta=1.0,
        center_freq_hz=50.0,
        gain_bandwidth_hz=10.0,
        mode_spacing_hz=0.2,
        sim_duration=2.0,
        points_per_sec=2000
):
    """
    Simulates the multimode laser dynamics based on equations S5-S9.
    """
    N = len(initial_population_inversion)
    # This check prevents running the simulation on empty projections
    if N == 0 or np.all(initial_population_inversion == 0):
        print("Skipping simulation for empty projection.")
        return {
            'time': np.linspace(0, sim_duration, int(sim_duration * points_per_sec)),
            'N_modes': N,
            'initial_inversion': initial_population_inversion,
            'final_inversion': np.zeros(N),
            'final_amplitude': np.zeros(N),
            'output_signal': np.zeros(int(sim_duration * points_per_sec)),
        }

    print(f"Running multimode laser simulation with {N} modes.")
    epsilon = 1e-12
    mu_indices = np.arange(1, N + 1)
    natural_frequencies_hz = center_freq_hz - 0.5 * (gain_bandwidth_hz - mode_spacing_hz * (2 * mu_indices - 1))
    omega_natural_rad = 2 * np.pi * natural_frequencies_hz

    d0 = initial_population_inversion
    A0 = np.random.uniform(0, 1e-4, N)
    phi0 = np.random.uniform(0, 2 * np.pi, N)
    y0 = np.concatenate([d0, A0, phi0])

    def laser_ode_system(t, y):
        d = y[0:N]
        A = y[N:2 * N]
        phi = y[2 * N:3 * N]

        X = np.sum(A * np.cos(phi))
        Y = np.sum(A * np.sin(phi))
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        sum_term_S6_and_S7_cos = X * cos_phi + Y * sin_phi
        sum_term_S6_and_S7_sin = Y * cos_phi - X * sin_phi

        d_d_dt = -d / T1 - 4 * coupling_beta * d * (sum_term_S6_and_S7_cos ** 2)
        d_A_dt = -A / T2 + coupling_beta * d * sum_term_S6_and_S7_cos
        d_phi_dt = omega_natural_rad + coupling_beta * (d / (A + epsilon)) * sum_term_S6_and_S7_sin
        dydt = np.concatenate([d_d_dt, d_A_dt, d_phi_dt])

        return dydt

    t_span = [0, sim_duration]
    sol = solve_ivp(laser_ode_system, t_span, y0, method='RK45', dense_output=True)
    output_time_points = np.linspace(t_span[0], t_span[1], int(sim_duration * points_per_sec))
    output_signal = np.zeros(len(output_time_points))
    final_state_y = sol.sol(output_time_points[-1])

    for i, t_point in enumerate(output_time_points):
        current_state_y = sol.sol(t_point)
        A_current = current_state_y[N:2 * N]
        phi_current = current_state_y[2 * N:3 * N]
        output_signal[i] = (1 / np.sqrt(N)) * np.sum(A_current * np.cos(phi_current))

    return {
        'time': output_time_points,
        'N_modes': N,
        'initial_inversion': d0,
        'final_inversion': final_state_y[0:N],
        'final_amplitude': final_state_y[N:2*N],
        'output_signal': output_signal,
    }

def plot_results(results, png_path):
    """
    Visualizes the simulation results from the multimode laser model.
    """
    time = results['time']
    N = results['N_modes']
    signal = results['output_signal']
    mode_indices = np.arange(1, N + 1) if N > 0 else []

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Multimode Laser Simulation (Single Projection)", fontsize=16)

    ax1 = axes[0, 0]
    if N > 0:
        ax1.plot(mode_indices, results['initial_inversion'], 'o-', label=f'$d_\\mu(t=0)$')
        ax1.plot(mode_indices, results['final_inversion'], 'o-', label=f'$d_\\mu(t={time[-1]:.2f}s)$')
    ax1.set_title("Population Inversion Profiles")
    ax1.set_xlabel("Mode Index ($\\mu$)")
    ax1.set_ylabel("Population Inversion ($d_\\mu$)")
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[0, 1]
    if N > 0:
        ax2.bar(mode_indices, results['final_amplitude'], width=0.8)
    ax2.set_title(f"Final Mode Amplitudes ($A_\\mu$) at t={time[-1]:.2f}s")
    ax2.set_xlabel("Mode Index ($\\mu$)")
    ax2.set_ylabel("Amplitude ($A_\\mu$)")
    ax2.grid(True, axis='y')

    ax3 = axes[1, 0]
    ax3.plot(time, signal)
    ax3.set_title("Total Output Signal $Sig(t)$ (Eq. S9)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Signal Amplitude (a.u.)")
    ax3.grid(True)

    min_freq_hz = 35
    max_freq_hz = 65
    ax4 = axes[1, 1]
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

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(png_path)
    plt.show()
    print(f"Saved results as PNG to {png_path}")
    plt.close(fig)

### NEW ###
def plot_reconstruction_comparison(original_image, reconstructed_image, path):
    """
    Plots the original and reconstructed images side-by-side for comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Image Reconstruction Comparison", fontsize=16)

    axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original Image")
    axes[0].set_axis_off()

    axes[1].imshow(reconstructed_image, cmap='gray')
    axes[1].set_title("Reconstructed from RASER Signals")
    axes[1].set_axis_off()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path)
    plt.show()
    print(f"Saved reconstruction comparison to {path}")
    plt.close(fig)

# --- Main execution block ---
if __name__ == '__main__':
    # --- Physical Constants & Experimental Parameters ---
    MU0, H_BAR, GAMMA_H = 4 * np.pi * 1e-7, 1.05457e-34, 2.67522e8
    T1_paper, T2_paper, Q_paper = 5.0, 0.7, 100
    Vs_paper, N_slices = 0.5 * (1e-2)**3, 50
    delta_nu_paper, Delta_paper, nu0_paper = 0.2, 10.0, 50.0
    d0_total = 25e16
    coupling_beta_calculated = (MU0 * H_BAR * GAMMA_H**2 * Q_paper) / (4 * Vs_paper)
    print(f"Calculated Coupling Constant (beta): {coupling_beta_calculated:.4e}")

    # --- Setup Output Directory ---
    output_directory_base = Path('./Sim_RASER_Output/') # Using a local directory
    model_name = '{}_{}'.format('Raser_Reconstruction', datetime.now().strftime("%Y%m%d-%H%M%S"))
    res_dir = output_directory_base / model_name
    makedirs(res_dir)
    print(f"Results will be saved in: {res_dir}")

    # --- Reconstruction Parameters ---
    N_PROJECTIONS = 60  # Number of angles to scan (e.g., 1 projection per degree)
    MIN_FREQ_HZ = 35
    MAX_FREQ_HZ = 65

    # 1. Create the original 2D image
    initial_map = create_sample_inversion_map(shape=(N_slices, N_slices))

    # 2. Set up angles and initialize the sinogram
    theta = np.linspace(0., 180., N_PROJECTIONS, endpoint=False)
    # Use `radon` to get the expected detector size for the sinogram
    # `circle=True` ensures the projection size is consistent for reconstruction
    placeholder_proj = radon(initial_map, theta=[0], circle=True)
    detector_len = len(placeholder_proj)
    reconstructed_sinogram = np.zeros((detector_len, N_PROJECTIONS))

    print(f"\nStarting reconstruction with {N_PROJECTIONS} projections.")
    print(f"Each reconstructed projection will be interpolated to a detector length of {detector_len}.")

    # --- Main Tomography Loop ---
    for i, angle in enumerate(theta):
        print(f"\n--- Processing Projection {i+1}/{N_PROJECTIONS} (Angle: {angle:.1f}°) ---")

        # A. Get the 1D projection for the current angle using Radon transform
        projection = radon(initial_map, theta=[angle], circle=True).flatten()

        # B. Prepare projection as input for simulation
        # Not sure how the normaliztion should work, I prefer by sum
        if np.max(projection) > 0:
            #projection = projection / np.max(projection)
            projection = projection / np.sum(projection)
        initial_population_inversion = d0_total * projection

        # C. Run the RASER simulation
        sim_results = simulate_laser_dynamics(
            initial_population_inversion=initial_population_inversion,
            T1=T1_paper, T2=T2_paper, coupling_beta=coupling_beta_calculated,
            center_freq_hz=nu0_paper, gain_bandwidth_hz=Delta_paper,
            mode_spacing_hz=delta_nu_paper, sim_duration=2.0, points_per_sec=2000
        )

        # D. Process the output signal to get the reconstructed 1D projection
        signal, time = sim_results['output_signal'], sim_results['time']
        dt = time[1] - time[0]
        signal_fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal_fft), d=dt)

        shifted_freqs = np.fft.fftshift(freqs)
        shifted_magnitudes = np.fft.fftshift(np.abs(signal_fft))
        freq_mask = (shifted_freqs >= MIN_FREQ_HZ) & (shifted_freqs <= MAX_FREQ_HZ)
        filtered_magnitudes = shifted_magnitudes[freq_mask]

        # E. Interpolate result to match the standard detector length
        if len(filtered_magnitudes) > 1:
            x_source = np.linspace(0, 1, len(filtered_magnitudes))
            x_target = np.linspace(0, 1, detector_len)
            reconstructed_projection_1d = np.interp(x_target, x_source, filtered_magnitudes)
        else:
            reconstructed_projection_1d = np.zeros(detector_len)

        # F. Store the result in our sinogram
        reconstructed_sinogram[:, i] = reconstructed_projection_1d

    print("\n--- All projections processed. Reconstructing 2D image... ---")

    # 3. Reconstruct the 2D image using the Inverse Radon Transform
    reconstructed_image = iradon(reconstructed_sinogram, theta=theta, filter_name='ramp', circle=True)

    # 4. Save artifacts
    np.save(res_dir / 'initial_2d_map.npy', initial_map)
    np.save(res_dir / 'reconstructed_sinogram.npy', reconstructed_sinogram)
    np.save(res_dir / 'reconstructed_image.npy', reconstructed_image)
    # Normalize and save the reconstructed image as a visual png
    rec_img_norm = (reconstructed_image - np.min(reconstructed_image)) / (np.max(reconstructed_image) - np.min(reconstructed_image))
    imsave(res_dir / 'reconstructed_image.png', (rec_img_norm * 255).astype(np.uint8))
    print("Saved final reconstruction artifacts.")

    # FIGURE 1: Plot results for one random projection
    print("\nGenerating Figure 1: Detailed analysis for a single random projection...")
    random_index = random.randint(0, N_PROJECTIONS - 1)
    random_angle = theta[random_index]
    print(f"Selected random angle for Figure 1: {random_angle:.1f}°")

    random_projection = radon(initial_map, theta=[random_angle], circle=True).flatten()
    if np.max(random_projection) > 0:
        random_projection /= np.max(random_projection)
    initial_pop_inv_random = d0_total * random_projection

    single_run_results = simulate_laser_dynamics(
        initial_population_inversion=initial_pop_inv_random,
        T1=T1_paper, T2=T2_paper, coupling_beta=coupling_beta_calculated,
        center_freq_hz=nu0_paper, gain_bandwidth_hz=Delta_paper,
        mode_spacing_hz=delta_nu_paper, sim_duration=2.0, points_per_sec=2000
    )
    plot_results(single_run_results, res_dir / 'Figure1_Single_Projection_Analysis.png')

    # FIGURE 2: Plot the original vs. reconstructed image
    print("\nGenerating Figure 2: Original vs. Reconstructed image comparison...")
    plot_reconstruction_comparison(initial_map, reconstructed_image, res_dir / 'Figure2_Reconstruction_Comparison.png')

    print("\n--- Script Finished ---")
