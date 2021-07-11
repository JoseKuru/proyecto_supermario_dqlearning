import flask, argparse, os, sys
import pandas as pd

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from utils_.apis_tb import comprobacion_token
from utils_.sql_tb import insert_table

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--X', type=str, help='Introduce el nombre del estudiante')
args = parser.parse_args()
if args.X == "josecarlos":

    print('Iniciando conexión')
    app = flask.Flask(__name__)

    @app.route('/json', methods=['GET'])
    def create_json():
        json_data = comprobacion_token()
        return json_data

    @app.route('/mysql', methods=['GET'])
    def mysql():
        df = pd.read_csv(os.path.dirname(src_path) + os.sep + 'reports' + os.sep + 'df.csv')
        nombre = 'jose_carlos_batista_rivero'
        insert_table(df, nombre)
        return '¡Tabla insertada correctamente!'

    app.run(host= 'localhost', port=5000, debug=True)

else:
    print('Nombre incorrecto')
