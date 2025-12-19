import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
import time
from dash import Dash, dcc, html, Input, Output, State, clientside_callback
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config

params = load_config('config.yml')

# --- System Parameters ---
Gamma = float(params['gamma'])  # Pumping rate
d1_0 = float(params['d1_0'])   # Starting value for d1
d2_0 = float(params['d2_0'])    # Starting  value for d2

# Define the system of 6 coupled ODEs with named variables
def system(t, y):
    d1, d2, a1, a2, phi1, phi2 = y

    T1 = float(params['T1'])
    T2 = float(params['T2'])
    coupling_beta = float(params['coupling_beta'])
    deltaNu = float(params['deltaNu'])  # Distance (in Hz) between peaks
    nu0 = float(params['nu0'])
    epsilon = float(params['epsilon'])
    f_off = float(params['f_off'])  # frequency offset to move both signals positive

    # The dd1_dt and dd2_dt equations have been updated to include a pumping term.
    dd1_dt = Gamma * (d1_0 - d1) - (d1 / T1) - 4 * coupling_beta * (a1**2 + a1 * a2 * np.cos(phi1 - phi2))
    dd2_dt = Gamma * (d2_0 - d2) - (d2 / T1) - 4 * coupling_beta * (a2**2 + a2 * a1 * np.cos(phi1 - phi2))

    da1_dt = -(a1 / T2) + coupling_beta * d1 * (a1 + a2 * np.cos(phi1 - phi2))
    da2_dt = -(a2 / T2) + coupling_beta * d2 * (a2 + a1 * np.cos(phi1 - phi2))

    dphi1_dt = 2 * np.pi * (nu0 + f_off + (deltaNu / 2)) + coupling_beta * (d1 / max(a1,epsilon)) * a2 * np.sin(phi2 - phi1)
    dphi2_dt = 2 * np.pi * (nu0 + f_off - (deltaNu / 2)) + coupling_beta * (d2 / max(a2,epsilon)) * a1 * np.sin(phi2 - phi1)

    return [dd1_dt, dd2_dt, da1_dt, da2_dt, dphi1_dt, dphi2_dt]

# Time span and initial conditions
t_f = 30
t_span = (0, t_f)
t_eval = np.linspace(*t_span, 200*t_f)
initial_conditions = [d1_0, d2_0, 1e10, 1e10, np.pi/2, np.pi/3]  # [d1, d2, a1, a2, phi1, phi2]

print("Starting ODE Solver...")
start = time.perf_counter()
solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, method='BDF')
end = time.perf_counter()
print("ODE Solver done")
print(f"{end - start:.4f} seconds elapsed")

t = solution.t
a1, a2, phi1, phi2 = solution.y[2], solution.y[3], solution.y[4], solution.y[5]
output_signal = (1 / np.sqrt(2)) * (a1 * np.real(np.exp(1j * phi1)) + a2 * np.real(np.exp(1j * phi2)))

# Pre-compute Global FFT
freq_full = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
Y_full = np.abs(np.fft.fft(output_signal))
pos_mask = freq_full > 0

# --- 2. Dash App Setup ---
# --- 3. Dash App ---
app = Dash(__name__)

app.layout = html.Div([
    # Add a specific ID for the content we want to print
    html.Div(id="printable-content", children=[
        html.H2("Interactive Signal Analysis Dashboard", style={'textAlign': 'center'}),

        dcc.Graph(id='time-plot'),
        html.Div(style={'height': '30px'}),
        dcc.Graph(id='freq-plot'),

        html.Div([
            html.Div([
                html.H4("System Parameters"),
                html.Table([
                    html.Tr([
                        html.Td(f"{k}:", style={'paddingRight': '15px', 'width': '120px', 'fontWeight': 'bold'}),
                        html.Td(v)
                    ]) for k, v in params.items()
                ], style={'width': 'auto', 'border': '1px solid #e1e1e1', 'padding': '10px',
                          'backgroundColor': 'white'})
            ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.H4("Run Notes & Observations"),
                dcc.Textarea(
                    id='run-notes',
                    placeholder='Enter observations here...',
                    style={'width': '100%', 'height': '150px'}
                ),
            ], style={'width': '50%', 'display': 'inline-block', 'float': 'right'})
        ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'marginTop': '20px', 'display': 'flex',
                  'justifyContent': 'space-between'})
    ]),

    # The Download Button (Hidden during print via CSS)
    html.Div([
        html.Button("Download Full Page (PDF)", id="btn-print",
                    style={'padding': '15px', 'fontSize': '16px', 'backgroundColor': '#007bff', 'color': 'white',
                           'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
    ], style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '50px'}, className="no-print")

], style={'padding': '20px'})

# --- CSS to hide the button during export ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @media print {
                .no-print {
                    display: none !important;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- Logic to trigger the browser's print dialog ---
clientside_callback(
    """
    function(n_clicks) {
        if(n_clicks > 0) {
            window.print();
        }
        return null;
    }
    """,
    Output("btn-print", "children"),
    Input("btn-print", "n_clicks")
)


@app.callback(
    [Output('time-plot', 'figure'),
     Output('freq-plot', 'figure')],
    [Input('time-plot', 'relayoutData'),
     Input('run-notes', 'value')]
)
def update_plots(relayoutData, run_notes):
    t_min, t_max = t[0], t[-1]
    is_zoomed = False

    if relayoutData and 'xaxis.range[0]' in relayoutData:
        t_min, t_max = relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
        is_zoomed = True

    indices = np.where((t >= t_min) & (t <= t_max))[0]

    # --- Time Plot ---
    fig_time = go.Figure(go.Scatter(x=t, y=output_signal, name="Signal", line=dict(color='royalblue')))
    if is_zoomed:
        fig_time.add_vrect(x0=t_min, x1=t_max, fillcolor="LightSalmon", opacity=0.3, layer="below")

    fig_time.update_layout(title="Time Domain", xaxis_title="Time", height=350, margin=dict(t=40, b=40))

    # --- Frequency Plot ---
    fig_freq = go.Figure()
    fig_freq.add_trace(
        go.Scatter(x=freq_full[pos_mask], y=Y_full[pos_mask], name="Full FFT", line=dict(color='lightgrey')))

    if len(indices) > 1:
        f_z = np.fft.fftfreq(len(indices), d=(t[1] - t[0]))
        Y_z = np.abs(np.fft.fft(output_signal[indices]))
        fig_freq.add_trace(
            go.Scatter(x=f_z[f_z > 0], y=Y_z[f_z > 0], name="Zoomed FFT", line=dict(color='red', dash='dash')))

    # Add Parameters and Run Notes as Plotly Annotations (for export)
    param_text = "<br>".join([f"{k}: {v}" for k, v in params.items()])
    notes_text = f"<b>Notes:</b> {run_notes}" if run_notes else "<b>Notes:</b> None"

    # --- Frequency Plot ---
    fig_freq = go.Figure()
    fig_freq.add_trace(
        go.Scatter(x=freq_full[pos_mask], y=Y_full[pos_mask], name="Full FFT", line=dict(color='lightgrey')))

    if len(indices) > 1:
        # Note: Using t[1]-t[0] assumes uniform sampling
        f_z = np.fft.fftfreq(len(indices), d=(t[1] - t[0]))
        Y_z = np.abs(np.fft.fft(output_signal[indices]))
        fig_freq.add_trace(
            go.Scatter(x=f_z[f_z > 0], y=Y_z[f_z > 0], name="Zoomed FFT", line=dict(color='red', dash='dash')))

    # Format parameters into two columns for better spacing
    param_list = [f"{k}: {v}" for k, v in params.items()]
    midpoint = len(param_list) // 2
    col1 = "<br>".join(param_list[:midpoint])
    col2 = "<br>".join(param_list[midpoint:])

    notes_text = f"<b>Notes:</b><br>{run_notes}" if run_notes else "<b>Notes:</b> None"

    fig_freq.update_layout(
        title="Frequency Domain Analysis",
        xaxis_title="Frequency (Hz)",
        xaxis_range=[0, 100],
        height=700,  # Increased height to accommodate text
        margin=dict(t=50, b=50, l=50, r=50),  # Increased bottom margin (b=200)
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig_time, fig_freq


if __name__ == '__main__':
    app.run(debug=True)