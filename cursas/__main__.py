import dash


from cursas.visualise import build_full_app 
import re 

#from scss.compiler import Compiler
#compiler = Compiler()
#
#convert = compiler.compile("assets/css/style.scss")
#
#with open("assets/style.css", "w") as out_file:
#    out_file.write("/* This is a machine generated file, changes may be overwritten */")
#    out_file.write(convert)

app = dash.Dash(
        title='Cursas',
        url_base_pathname='/cursas/',
        assets_folder='../assets',
        assets_ignore=re.escape('*.scss')
        )
app = build_full_app(app)
application = app.server

app.run_server(debug=True, use_reloader=True)  
#from cursas.extract import generate_full_run_table
#generate_full_run_table('eastville')
