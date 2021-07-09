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

from utils.dashboard_tb import bienvenida

add_selectbox = st.sidebar.selectbox('Menu', options=['Bienvenida', 'Planetas potencialmente habitables', 'Flask'])

if add_selectbox == 'Bienvenida':
    bienvenida()

if add_selectbox == 'Planetas potencialmente habitables':
    grafico_ESI()

if add_selectbox == 'Flask':
    dataframe()
