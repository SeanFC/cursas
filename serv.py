import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

#app = dash.Dash(__name__, requests_pathname_prefix='/')

#import os 
#print(os.environ['DASH_REQUESTS_PATHNAME_PREFIX'])
app = dash.Dash(
        __name__,
        #serve_locally = False, 
        title='Cursus',
        #requests_pathname_prefix='/myapp/',
        #routes_pathname_prefix='/',
        url_base_pathname='/myapp/',
        #include_assets_files=False
        )
app.layout = html.Div([
    #html.P("Color:"),
    #dcc.Dropdown(
    #    id="dropdown",
    #    options=[
    #        {'label': x, 'value': x}
    #        for x in ['Gold', 'MediumTurquoise', 'LightGreen']
    #    ],
    #    value='Gold',
    #    clearable=False,
    #),
    dcc.Graph(id="graph"),
])

application = app.server
#def application(envio, resp):
#    print(envio)
#    return app.server(envio, resp)

if __name__ == "__main__":
    app.run_server(debug=False, port=3031)

#if __name__ == '__main__':
#    from flup.server.scgi import WSGIServer
#    WSGIServer(application).run()
