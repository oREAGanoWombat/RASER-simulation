from dash import Dash, dcc, html, Input, Output, State, clientside_callback, callback_context

# --- 2. Dash App Setup ---
# --- 3. Dash App ---
app = Dash(__name__)

app.layout = html.Div([
    # Store to hold multiple highlight regions
    dcc.Store(id='selections-store', data=[]),

    # Add a specific ID for the content we want to print
    html.Div(id="printable-content", children=[
        html.H2("Interactive Signal Analysis Dashboard", style={'textAlign': 'center'}),

        # Added Clear Button
        html.Div([
            html.Button("Clear Selections", id="clear-btn",
                        style={'padding': '8px 15px', 'backgroundColor': '#dc3545',
                               'color': 'white', 'border': 'none', 'borderRadius': '4px',
                               'cursor': 'pointer', 'marginBottom': '10px'})
        ], style={'textAlign': 'right'}, className="no-print"),

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
     Output('freq-plot', 'figure'),
     Output('selections-store', 'data')],
    [Input('time-plot', 'relayoutData'),
     Input('clear-btn', 'n_clicks'),
     Input('run-notes', 'value')],
    [State('selections-store', 'data')]
)
def update_plots(relayoutData, clear_clicks, run_notes, selections):
    # Determine which component triggered the callback
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Initialize selections if empty
    if selections is None:
        selections = []

    # Handle clearing selections
    if triggered_id == 'clear-btn':
        selections = []
    # If the user zoomed on the time plot, append the new region
    elif triggered_id == 'time-plot' and relayoutData:
        if 'xaxis.range[0]' in relayoutData:
            t_min = relayoutData['xaxis.range[0]']
            t_max = relayoutData['xaxis.range[1]']
            # Prevent adding the exact same region twice sequentially
            if not selections or selections[-1] != [t_min, t_max]:
                selections.append([t_min, t_max])
        # Allow double-click to also reset/clear the selections
        elif 'xaxis.autorange' in relayoutData:
            selections = []

    # --- Time Plot ---
    fig_time = go.Figure(go.Scatter(x=t, y=output_signal, name="Signal", line=dict(color='royalblue')))

    # Draw ALL stored selections
    for sel in selections:
        fig_time.add_vrect(x0=sel[0], x1=sel[1], fillcolor="LightSalmon", opacity=0.3, layer="below")

    fig_time.update_layout(title="Time Domain", xaxis_title="Time", height=350, margin=dict(t=40, b=40))

    # --- Frequency Plot ---
    fig_freq = go.Figure()
    fig_freq.add_trace(
        go.Scatter(x=freq_full[pos_mask], y=Y_full[pos_mask], name="Full FFT", line=dict(color='lightgrey')))

    # Cycle through colors for different zoomed sections
    colors = ['red', 'green', 'orange', 'purple', 'cyan', 'magenta']

    for i, sel in enumerate(selections):
        t_min, t_max = sel
        indices = np.where((t >= t_min) & (t <= t_max))[0]

        if len(indices) > 1:
            # Switched to rfft and rfftfreq to align with global FFT calculations
            f_z = np.fft.rfftfreq(len(indices), d=(t[1] - t[0]))
            Y_z = np.abs(np.fft.rfft(output_signal[indices]))
            color = colors[i % len(colors)]

            fig_freq.add_trace(
                go.Scatter(x=f_z[f_z > 0], y=Y_z[f_z > 0], name=f"Zoomed FFT {i + 1}",
                           line=dict(color=color, dash='dash'))
            )

    # Add Parameters and Run Notes as Plotly Annotations
    param_list = [f"{k}: {v}" for k, v in params.items()]
    midpoint = len(param_list) // 2
    col1 = "<br>".join(param_list[:midpoint])
    col2 = "<br>".join(param_list[midpoint:])

    notes_text = f"<b>Notes:</b><br>{run_notes}" if run_notes else "<b>Notes:</b> None"

    fig_freq.update_layout(
        title="Frequency Domain Analysis",
        xaxis_title="Frequency (Hz)",
        xaxis_range=[0, 100],
        height=700,
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    # Notice we now return 'selections' to save the state back to the dcc.Store
    return fig_time, fig_freq, selections


if __name__ == '__main__':
    app.run(debug=True)