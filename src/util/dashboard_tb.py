import numpy as np
import pandas as pd
import streamlit as st
import os, requests
import plotly.express as px

def bienvenida():
    st.title('Entrenamiento reforzado con Super Mario')
    st.write('En 2009 la NASA lanzó el telescopio Kepler, el mayor proyecto hasta la fecha para el descubrimiento de exoplanetas. Despues de 10 años, unos 4.000 planetas han sido confirmados, y una pregunta frecuente en la ciencia ficción empezo a tener sus primeras evidencias')
    st.write('¿Habra planetas con condiciones similares a la tierra para ser colonozidas?')
