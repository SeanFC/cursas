""" Build the project as a wsgi application """

# External imports
import dash

# Internal imports
from cursas.app import build_full_app
from cursas.config import CursasConfig

# Build application
app = dash.Dash(
        title='Cursas',
        url_base_pathname='/cursas/',
        )
app = build_full_app(app, CursasConfig())

application = app.server # Hook for WSGI application
