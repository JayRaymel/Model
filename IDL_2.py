import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Simulated dataset for demonstration purposes
data = pd.DataFrame({
    'Feature_1': range(1, 101),
    'Feature_2': [x * 2 for x in range(1, 101)],
    'Predicted_Value': [x * 3 + 10 for x in range(1, 101)]
})

# Create the Dash app
app = dash.Dash(__name__)
app.title = "Predictive Model Dashboard"

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Predictive Model Results Dashboard", style={'textAlign': 'center'}),

    # Filters and controls
    html.Div([
        html.Label("Select Feature for X-axis:"),
        dcc.Dropdown(
            id='x-axis-feature',
            options=[
                {'label': 'Feature 1', 'value': 'Feature_1'},
                {'label': 'Feature 2', 'value': 'Feature_2'}
            ],
            value='Feature_1',
            clearable=False
        ),
        
        html.Label("Select Range of Predicted Values:"),
        dcc.RangeSlider(
            id='predicted-range-slider',
            min=data['Predicted_Value'].min(),
            max=data['Predicted_Value'].max(),
            step=1,
            marks={int(i): str(i) for i in range(int(data['Predicted_Value'].min()), int(data['Predicted_Value'].max()), 10)},
            value=[data['Predicted_Value'].min(), data['Predicted_Value'].max()]
        )
    ], style={'padding': '10px'}),

    # Graph for visualizing the results
    dcc.Graph(id='results-graph'),

    # Analysis and insights section
    html.Div([
        html.H2("Analysis and Insights", style={'textAlign': 'center'}),
        html.P("This section provides an analysis of the model's predictions and key insights derived from the data.",
               style={'textAlign': 'justify'}),
        html.Ul([
            html.Li("The model shows a strong correlation between Feature 1 and the Predicted Value."),
            html.Li("The Predicted Value increases linearly with both Feature 1 and Feature 2."),
            html.Li("Interactive filtering helps identify trends in specific ranges of predicted values.")
        ])
    ], style={'padding': '20px'}),

    # Conclusions and recommendations
    html.Div([
        html.H2("Conclusions and Recommendations", style={'textAlign': 'center'}),
        html.P("Based on the visualized data, we recommend further investigation into specific data points where the model predictions deviate from expected trends.",
               style={'textAlign': 'justify'})
    ])
])

# Callback to update the graph based on user input
@app.callback(
    Output('results-graph', 'figure'),
    [
        Input('x-axis-feature', 'value'),
        Input('predicted-range-slider', 'value')
    ]
)
def update_graph(x_feature, predicted_range):
    filtered_data = data[(data['Predicted_Value'] >= predicted_range[0]) & (data['Predicted_Value'] <= predicted_range[1])]
    fig = px.scatter(
        filtered_data, 
        x=x_feature, 
        y='Predicted_Value', 
        title=f"Predicted Value vs {x_feature}",
        labels={x_feature: x_feature, 'Predicted_Value': 'Predicted Value'},
        template="plotly_white"
    )
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)

