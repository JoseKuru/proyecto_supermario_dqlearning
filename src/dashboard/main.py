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

from utils_.dashboard_tb import bienvenida
from utils_.dashboard_tb import show_video

add_selectbox = st.sidebar.selectbox('Menu', options=['Bienvenida', 'Planetas potencialmente habitables', 'Flask', 'Demo'])

if add_selectbox == 'Bienvenida':
    bienvenida()

if add_selectbox == 'Flask':
    dataframe()

if add_selectbox == 'Demo':
    show_video()
