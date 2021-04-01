import dash

from cursas.app import build_full_app

from cursas.config import CursasConfig 

app = dash.Dash(
        title='Cursas',
        url_base_pathname='/cursas/',
        #assets_folder='../assets'
        )

app = build_full_app(app, CursasConfig())
application = app.server
