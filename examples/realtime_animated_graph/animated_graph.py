# ref: https://discuss.streamlit.io/t/how-to-access-uploaded-video-in-streamlit-by-open-cv/5831/7

import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import altair as alt
import cv2

st.title("Streamlit Demo")

video = 'Casino.Royale.2006__#01-46-40_01-47-14_label_B6-0-0.mp4'
video_dir = '../../../videos/'
output = './output'

csv = pd.read_csv(os.path.join(output, f'{video}.csv'))
del csv['video']
csv['index'] = [i for i in range(csv.shape[0])]

df = csv[['index', 'off']]

lines = alt.Chart(df).mark_line().encode(
    x=alt.X('1:Q', axis=alt.Axis(title='frames')),
    y=alt.Y('0:Q', axis=alt.Axis(title='off'))
).properties(
    width=640,
    height=300
)

def plot_animation(df):
    lines = alt.Chart(df).mark_line().encode(
       x=alt.X('index:Q', axis=alt.Axis(title='frames')),
       y=alt.Y('off:Q',axis=alt.Axis(title='off')),
    ).properties(
       width=640,
       height=300
    ) 
    return lines

N = df.shape[0]

line_plot = st.altair_chart(lines)
start_btn = st.button('Start')

i = 0

if start_btn:

    video_file = cv2.VideoCapture(os.path.join(video_dir, video))

    stframe = st.empty()

    while video_file.isOpened():
        #time.sleep(1/24)
        ret, frame = video_file.read()

        if not ret:
            print('can\'t receive frame. exiting...')
            break
    
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb)

        step_df = df.iloc[0:i]
        lines = plot_animation(step_df)
        line_plot = line_plot.altair_chart(lines)

        i += 1