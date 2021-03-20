import sys
import re 

import dash


from cursas.visualise import build_full_app, build_dev_app
from cursas.extract import pull_db
from cursas.config import get_config

#from scss.compiler import Compiler
#compiler = Compiler()
#
#convert = compiler.compile("assets/css/style.scss")
#
#with open("assets/style.css", "w") as out_file:
#    out_file.write("/* This is a machine generated file, changes may be overwritten */")
#    out_file.write(convert)


if (len(sys.argv) > 1) and (sys.argv[1] == 'pull'):
    pull_db(get_config())
    exit()

app = dash.Dash(
        title='Cursas',
        url_base_pathname='/cursas/',
        assets_folder='../assets',
        assets_ignore=re.escape('*.scss'),
        #assets_external_path='https://www.sfcleator.com/assets'
        meta_tags =[dict(name="viewport", content="width=device-width, initial-scale=1.0")]
        )


if (len(sys.argv) > 1) and (sys.argv[1] == 'dev'):
    app = build_dev_app(app, get_config())
else:
    app = build_full_app(app, get_config())

application = app.server
app.run_server(debug=True, use_reloader=True)  
#from cursas.extract import generate_full_run_table
#generate_full_run_table('eastville')
