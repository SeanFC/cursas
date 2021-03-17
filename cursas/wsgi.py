import dash

from cursas.visualise import build_full_app 

from cursas.config import get_config

app = dash.Dash(
        title='Cursas',
        url_base_pathname='/cursas/',
        #assets_folder='../assets'
        )

app = build_full_app(app, get_config())
application = app.server
