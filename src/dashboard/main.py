import os
import numpy as np
import pandas as pd
import streamlit as st
import os, sys, requests
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
path = os.path.abspath(__file__)
for i in range(2):
    path = os.path.dirname(path)
sys.path.append(path)

from utils_.dashboard_tb import bienvenida, show_video, show_graphics
from utils_.sql_tb import do_query

add_selectbox = st.sidebar.selectbox('Menu', options=['Bienvenida', 'Graficos', 'Flask', 'SQL', 'Demo'])

if add_selectbox == 'Bienvenida':
    bienvenida()

if add_selectbox == 'Graficos':
    show_graphics()

if add_selectbox == 'Flask':
    st.write(requests.get('http://localhost:5000/json?tokenize_id=B49078469').json())

if add_selectbox == 'SQL':
    query = """SELECT * FROM jose_carlos_batista_rivero"""
    df = do_query(query)
    st.table(df)

if add_selectbox == 'Demo':
    show_video()
