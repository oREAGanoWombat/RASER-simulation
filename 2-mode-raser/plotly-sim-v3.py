import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks, peak_widths  # <-- Added for measuring features
import time
from dash import Dash, dcc, html, Input, Output, State, clientside_callback, callback_context
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config


params = load_config('config.yml')

# --- System Parameters ---
Gamma = float(params['gamma'])
init_d1 = float(params['init_d1'])
init_d2 = float(params['init_d2'])


def system(t, y):
    d1, d2, a1, a2, phi1, phi2 = y
    T1 = float(params['T1'])
    T2 = float(params['T2'])
    d1_0 = float(params['d1_0'])
    d2_0 = float(params['d2_0'])
    coupling_beta = float(params['coupling_beta'])
    deltaNu = float(params['deltaNu'])
    nu0 = float(params['nu0'])
    epsilon = float(params['epsilon'])
    f_off = float(params['f_off'])

    dd1_dt = Gamma * (d1_0 - d1) - (d1 / T1) - (4 * coupling_beta) * (a1 ** 2 + a1 * a2 * np.cos(phi2 - phi1))
    dd2_dt = Gamma * (d2_0 - d2) - (d2 / T1) - (4 * coupling_beta) * (a2 ** 2 + a2 * a1 * np.cos(phi2 - phi1))
    da1_dt = -(a1 / T2) + coupling_beta * d1 * (a1 + a2 * np.cos(phi2 - phi1))
    da2_dt = -(a2 / T2) + coupling_beta * d2 * (a2 + a1 * np.cos(phi2 - phi1))
    dphi1_dt = 2 * np.pi * (nu0 + f_off + (deltaNu / 2)) + coupling_beta * (d1 / max(a1, epsilon)) * a2 * np.sin(
        phi2 - phi1)
    dphi2_dt = 2 * np.pi * (nu0 + f_off - (deltaNu / 2)) + coupling_beta * (d2 / max(a2, epsilon)) * a1 * np.sin(
        phi1 - phi2)

    return [dd1_dt, dd2_dt, da1_dt, da2_dt, dphi1_dt, dphi2_dt]


t_span = (0, 60)
t_eval = np.linspace(*t_span, 200 * 60)
initial_conditions = [init_d1, init_d2, 1e10, 1e10, np.pi / 2, np.pi / 3]

print("Starting ODE Solver...")
solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, method='BDF')
print("ODE Solver done")

t = solution.t
a1, a2, phi1, phi2 = solution.y[2], solution.y[3], solution.y[4], solution.y[5]
output_signal = (1 / np.sqrt(2)) * (a1 * np.real(np.exp(1j * phi1)) + a2 * np.real(np.exp(1j * phi2)))

freq_full = np.fft.rfftfreq(len(t), d=(t[1] - t[0]))
Y_full = np.abs(np.fft.rfft(output_signal))
pos_mask = freq_full > 0

# --- Dash App Setup ---
app = Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='selections-store', data=[]),

    html.Div(id="printable-content", children=[
        html.H2("Interactive Signal Analysis Dashboard", style={'textAlign': 'center'}),

        # Control Panel Banner
        html.Div([
            html.Div([
                html.Label("Layout: ", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.RadioItems(
                    id='layout-toggle',
                    options=[
                        {'label': ' Stacked ', 'value': 'stacked'},
                        {'label': ' Side-by-Side ', 'value': 'side-by-side'}
                    ],
                    value='stacked',
                    inline=True,
                    style={'display': 'inline-block'}
                ),

                # NEW: Measuring Tool Toggle
                dcc.Checklist(
                    id='measuring-tool-toggle',
                    options=[{'label': ' Enable Peak Measuring Tool', 'value': 'active'}],
                    value=[],
                    inline=True,
                    style={'display': 'inline-block', 'marginLeft': '40px', 'fontWeight': 'bold', 'color': '#802222'}
                )
            ], style={'display': 'inline-block', 'float': 'left'}),

            html.Button("Clear Selections", id="clear-btn",
                        style={'padding': '8px 15px', 'backgroundColor': '#dc3545',
                               'color': 'white', 'border': 'none', 'borderRadius': '4px',
                               'cursor': 'pointer'})
        ], style={'textAlign': 'right', 'marginBottom': '15px'}, className="no-print"),

        # Graphs Flex Container
        html.Div(id='graphs-container', children=[
            html.Div(id='time-plot-wrapper', children=[dcc.Graph(id='time-plot')]),
            html.Div(id='freq-plot-wrapper', children=[dcc.Graph(id='freq-plot')])
        ]),

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
                dcc.Textarea(id='run-notes', placeholder='Enter observations here...',
                             style={'width': '100%', 'height': '150px'}),
            ], style={'width': '50%', 'display': 'inline-block', 'float': 'right'})
        ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'marginTop': '20px', 'display': 'flex',
                  'justifyContent': 'space-between'})
    ]),

    # The Download Button
        html.Div([
            html.Button("Download Full Page (PDF)", id="btn-print",
                        style={'padding': '15px', 'fontSize': '16px', 'backgroundColor': '#007bff', 'color': 'white',
                               'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
        ], style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '50px'}, className="no-print")

], style={'padding': '20px'})

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
        <style>
            @media print { .no-print { display: none !important; } }
            #time-plot-wrapper::-webkit-resizer { background-color: #6c757d; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}{%scripts%}{%renderer%}
            <script>
                const resizeObserver = new ResizeObserver(() => { window.dispatchEvent(new Event('resize')); });
                const checkExist = setInterval(function() {
                   const wrapper = document.getElementById('time-plot-wrapper');
                   if (wrapper) { resizeObserver.observe(wrapper); clearInterval(checkExist); }
                }, 500);
            </script>
        </footer>
    </body>
</html>
'''


@app.callback(
    [Output('graphs-container', 'style'), Output('time-plot-wrapper', 'style'), Output('freq-plot-wrapper', 'style')],
    [Input('layout-toggle', 'value')]
)
def update_layout_orientation(orientation):
    if orientation == 'side-by-side':
        return ({'display': 'flex', 'flexDirection': 'row', 'width': '100%'},
                {'resize': 'horizontal', 'overflow': 'hidden', 'width': '50%', 'minWidth': '20%', 'maxWidth': '80%',
                 'borderRight': '5px solid #ccc', 'paddingRight': '10px'},
                {'flex': '1', 'overflow': 'hidden', 'paddingLeft': '10px'})
    return ({'display': 'flex', 'flexDirection': 'column', 'width': '100%'}, {'width': '100%', 'marginBottom': '30px'},
            {'width': '100%'})


@app.callback(
    [Output('time-plot', 'figure'), Output('freq-plot', 'figure'), Output('selections-store', 'data')],
    [Input('time-plot', 'relayoutData'), Input('clear-btn', 'n_clicks'), Input('run-notes', 'value'),
     Input('layout-toggle', 'value'), Input('measuring-tool-toggle', 'value')],  # <-- Added tool input
    [State('selections-store', 'data')]
)
def update_plots(relayoutData, clear_clicks, run_notes, layout_mode, measuring_tool, selections):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if selections is None: selections = []

    if triggered_id == 'clear-btn':
        selections = []
    elif triggered_id == 'time-plot' and relayoutData:
        if 'xaxis.range[0]' in relayoutData:
            t_min, t_max = relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']
            if not selections or selections[-1] != [t_min, t_max]: selections.append([t_min, t_max])
        elif 'xaxis.autorange' in relayoutData:
            selections = []

    colors = ['red', 'green', 'orange', 'purple', 'cyan', 'magenta']
    time_height = 550 if layout_mode == 'side-by-side' else 350
    freq_height = 550 if layout_mode == 'side-by-side' else 700

    # --- Time Plot ---
    fig_time = go.Figure(go.Scatter(x=t, y=output_signal, name="Signal", line=dict(color='royalblue')))
    for i, sel in enumerate(selections):
        fig_time.add_vrect(x0=sel[0], x1=sel[1], fillcolor=colors[i % len(colors)], opacity=0.2, layer="below",
                           line_width=0)
    fig_time.update_layout(title="Time Domain", xaxis_title="Time", height=time_height, margin=dict(t=40, b=40))

    # --- Frequency Plot ---
    fig_freq = go.Figure()
    fig_freq.add_trace(
        go.Scatter(x=freq_full[pos_mask], y=Y_full[pos_mask], name="Full FFT", line=dict(color='lightgrey')))

    for i, sel in enumerate(selections):
        t_min, t_max = sel
        indices = np.where((t >= t_min) & (t <= t_max))[0]
        if len(indices) > 1:
            f_z = np.fft.rfftfreq(len(indices), d=(t[1] - t[0]))
            Y_z = np.abs(np.fft.rfft(output_signal[indices]))
            fig_freq.add_trace(go.Scatter(x=f_z[f_z > 0], y=Y_z[f_z > 0], name=f"Zoomed FFT {i + 1}",
                                          line=dict(color=colors[i % len(colors)], dash='dash')))

    # --- NEW: Measuring Tool Implementation ---
    if 'active' in measuring_tool:
        # Constrain search window to the visible 0-100Hz layout viewport
        mask_visible = (freq_full >= 0) & (freq_full <= 100)
        f_v = freq_full[mask_visible]
        Y_v = Y_full[mask_visible]

        # Identify local peaks matching a minimum threshold prominence
        peaks, _ = find_peaks(Y_v, prominence=np.max(Y_v) * 0.05)

        if len(peaks) >= 2:
            # Grab the 2 highest peaks and order them sequentially by frequency
            top_peaks = np.sort(peaks[np.argsort(Y_v[peaks])][::-1][:2])

            f1, f2 = f_v[top_peaks[0]], f_v[top_peaks[1]]
            y1, y2 = Y_v[top_peaks[0]], Y_v[top_peaks[1]]

            # 1. Delta Measurement Bar (Horizontal span line)
            y_bar = min(y1, y2) * 0.75
            fig_freq.add_shape(
                type="line", x0=f1, x1=f2, y0=y_bar, y1=y_bar,
                line=dict(color="#7f7f7f", width=1.5, dash="solid")
            )
            # Center delta metric text label
            fig_freq.add_annotation(
                x=(f1 + f2) / 2, y=y_bar, text=f"{abs(f2 - f1):.3f} Hz",
                showarrow=False, yshift=12, font=dict(color="#555555", size=13, weight="bold"),
                bgcolor="white", bordercolor="#7f7f7f", borderpad=2
            )

            # 2. Individual Peak FWHM Width Measurement
            widths, width_heights, left_ips, right_ips = peak_widths(Y_v, top_peaks, rel_height=0.5)
            hz_per_index = f_v[1] - f_v[0]

            for idx, p_idx in enumerate(top_peaks):
                peak_freq = f_v[p_idx]
                peak_amp = Y_v[p_idx]
                fwhm_width_hz = widths[idx] * hz_per_index

                # Generate specific layout labels targeting individual peak widths
                fig_freq.add_annotation(
                    x=peak_freq, y=peak_amp,
                    text=f"{fwhm_width_hz:.3f} Hz",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="maroon",
                    ax=-45 if idx == 0 else 45, ay=-40,
                    font=dict(color="maroon", size=11),
                    bordercolor="maroon", borderwidth=1, borderpad=3, bgcolor="#fffbfb"
                )

    fig_freq.update_layout(
        title="Frequency Domain Analysis", xaxis_title="Frequency (Hz)", xaxis_range=[0, 100],
        height=freq_height, margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig_time, fig_freq, selections


if __name__ == '__main__':
    app.run(debug=True)