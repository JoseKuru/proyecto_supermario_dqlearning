import sys, os
import numpy as np
import pandas as pd
import streamlit as st
import os, requests
import plotly.express as px

path_recordings = os.path.abspath(__file__)
for i in range(3):
    path_recordings = os.path.dirname(path_recordings)


def bienvenida():
    st.title('Entrenamiento reforzado con Super Mario')
    st.write('¿Será capaz la red neuronal de derrotar a Bowser?')
    st.write('Aquí mostraremos el resultado de aplicar un algoritmo de aprendizaje supervisado al videojuego Super Mario Bros.')

def show_video():
    st.title('Demo del modelo en acción')
    video = open(path_recordings + os.sep + 'reports' + os.sep +
    'recording' + os.sep + 'mario_edited.mp4', 'rb')
    video_bytes = video.read()
    st.video(video_bytes)