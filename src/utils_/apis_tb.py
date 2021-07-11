import flask
import pandas as pd
import os

path_project = os.path.abspath(__file__)
for i in range(3):
    path_project = os.path.dirname(path_project)

from utils_.sql_tb import insert_table

def comprobacion_token():
    '''Comprueba que el token es correcto'''
    token = flask.request.args.get('tokenize_id')
    if token == 'B49078469':
        data = pd.read_csv(path_project + os.sep + 'reports' + os.sep + 'df.csv').to_json(indent=4)
        return data

    return 'Token incorrecto'

