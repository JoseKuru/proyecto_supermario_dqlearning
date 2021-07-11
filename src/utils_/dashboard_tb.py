import sys, os
import numpy as np
import pandas as pd
import streamlit as st
import os, requests
import plotly.express as px
import pickle

path_project = os.path.abspath(__file__)
for i in range(3):
    path_project = os.path.dirname(path_project)

def bienvenida():
    st.title('Entrenamiento reforzado con Super Mario')
    st.write('¿Será capaz la red neuronal de derrotar a Bowser?')
    st.write('Aquí mostraremos el resultado de aplicar un algoritmo de aprendizaje supervisado al videojuego Super Mario Bros.')

def show_video():
    st.title('Demo del modelo en acción')
    video = open(path_project + os.sep + 'reports' + os.sep +
    'recording' + os.sep + 'mario_edited.mp4', 'rb')
    video_bytes = video.read()
    st.video(video_bytes)

def show_graphics():
    score_list = pickle.load(open(path_project + os.sep + 'reports' + os.sep + 'score_list.h5', 'rb'))
    average_score = [sum(score_list[x: x+10])/10 for x in range(0, len(score_list), 10)]
    fig = px.line(x=[x for x in range(len(average_score))], y=average_score, title='Average score por cada 10 episodios')
    st.plotly_chart(fig)