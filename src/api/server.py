import flask, argparse, os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.apis_tb import comprobacion_token, mysql_table, others

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--X', type=str, help='Introduce el nombre del estudiante')
args = parser.parse_args()
if args.X == "Jose_Carlos":

    print('Iniciando conexión')
    app = flask.Flask(__name__)

    @app.route('/json', methods=['GET'])
    def create_json():
        
        return comprobacion_token() # Devuelve un json

    @app.route('/mysql', methods=['GET'])
    def mysql():
        
        pass # Función que introduxa 

    @app.route('/others', methods=['GET'])
    def others():
        pass

    app.run(host= '127.0.0.1', port=5000, debug=True)

else:
    print('Nombre incorrecta')
