import flask
import pandas as pd
import os

def comprobacion_token():
    '''Comprueba que el token es correcto'''
    token = flask.request.args.get('tokenize_id')
    if token == 'B49078469':
        pass # Aqui se devolveria lo que uno pida

    return 'Token incorrecto'

def mysql_table():
    pass

def others():
    pass

