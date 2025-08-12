# Requirements
import numpy as np
import matplotlib.pyplot as plt

# plot_results() visualizes the simulation results from teh multimode RASER model
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
