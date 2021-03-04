import dash

from cursas.visualise import build_full_app 

app = dash.Dash(
        title='Cursas',
        url_base_pathname='/cursas/',
        )
app = build_full_app(app)
application = app.server
