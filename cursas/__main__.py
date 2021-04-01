""" Main entry point for running Cursas from the command line. """

# Standard library imports
import sys
import re

# External imports
import dash

# Internal imports
from cursas.app import build_full_app, build_dev_app
from cursas.extract import pull_db
from cursas.config import CursasConfig

# Pull the database
if (len(sys.argv) > 1) and (sys.argv[1] == 'pull'):
    pull_db(CursasConfig())
    sys.exit()

# Create the application
app = dash.Dash(
        title='Cursas',
        url_base_pathname='/cursas/',
        assets_folder='../assets',
        assets_ignore=re.escape('*.scss'),
        meta_tags =[dict(name="viewport", content="width=device-width, initial-scale=1.0")]
        )

# Add everything into the application
if (len(sys.argv) > 1) and (sys.argv[1] == 'dev'):
    app = build_dev_app(app, CursasConfig())
else:
    app = build_full_app(app, CursasConfig())

# Run the application and leave WSGI hook
application = app.server # WSGI hook
app.run_server(debug=True, use_reloader=True)
