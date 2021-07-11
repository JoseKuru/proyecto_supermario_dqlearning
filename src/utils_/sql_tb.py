import os, sys
import json
import pandas as pd
from pandas.io.sql import read_sql
from sqlalchemy import create_engine

def insert_table(df, nombre_tabla):
    
    dir = os.path.dirname
    ruta_archivo= os.path.abspath(__file__)

    with open(dir(ruta_archivo) + os.sep + 'sql_parameters.json', 'r') as data:
        sql_data = json.load(data)

    IP_DNS = sql_data['IP_DNS']
    PORT = str(sql_data['PORT'])
    USER = sql_data['USER']
    PASSWORD = sql_data['PASSWORD']
    BD_NAME = sql_data['BD_NAME']

    engine = create_engine('mysql+pymysql://' + USER + ':' + PASSWORD + '@' + IP_DNS +
            ':' + PORT + '/' + BD_NAME)
    df.to_sql(nombre_tabla ,con=engine, index=False)

def do_query(query):
    
    dir = os.path.dirname
    ruta_archivo= os.path.abspath(__file__)

    with open(dir(ruta_archivo) + os.sep + 'sql_parameters.json', 'r') as data:
        sql_data = json.load(data)

    IP_DNS = sql_data['IP_DNS']
    PORT = str(sql_data['PORT'])
    USER = sql_data['USER']
    PASSWORD = sql_data['PASSWORD']
    BD_NAME = sql_data['BD_NAME']

    engine = create_engine('mysql+pymysql://' + USER + ':' + PASSWORD + '@' + IP_DNS +
            ':' + PORT + '/' + BD_NAME)
    df = pd.read_sql(query, engine)
    
    return df

