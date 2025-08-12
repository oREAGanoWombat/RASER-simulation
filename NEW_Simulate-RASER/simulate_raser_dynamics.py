# Requirements
import numpy as np

def simulate_raser_dynamics(
        initial_population_inversion,
        T1=5.0,
        T2=0.7,
        coupling_beta=1.0,
        center_freq_hz=50.0,
        gain_bandwidth_hz=10.0,
        mode_spacing_hz=0.2,
        sim_duration=10.0,
        points_per_sec=2000,
        solver_method='BDF',
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
    mu_indices = np.arange(1, N + 1)
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
    sol = solve_ivp(raser_ode_system, t_span, y0, method=solver_method, dense_output=True)

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