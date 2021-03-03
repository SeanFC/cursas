import dash

from cursas.visualise import plot_female_21_24

app = dash.Dash(
        title='Cursas',
        url_base_pathname='/cursas/',
        )
app = plot_female_21_24(app)
application = app.server
