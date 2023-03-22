# # Generate dataset
# # Generate dataset
#需要设置间隔freq!!
# import matplotlib
# matplotlib.use('qt5Agg')  #!!! 指定后端
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from statistics import mode
import matplotlib
matplotlib.use('Agg')
import plotly.graph_objects as go
from numpy import *
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import math
import random   #test 1/5的测试集
import matplotlib.pyplot as plt
import streamlit as st
def draw_pic(jg) :

    fig= go.Figure(data=[go.Histogram(x=jg, histnorm='probability')])

    fig.update_layout(
                      xaxis_showgrid=False,
                      yaxis_showgrid=False,

                      title_text='Time Interval Distribution',
    showlegend = False)
    fig.update_xaxes(title_text='Time Interval（min)')
    fig.update_yaxes(title_text='Frequency')

    st.plotly_chart(fig, use_container_width=True)

    #plt.show()

    #st.pyplot(fig2)
def timeiter_fill_draw(df_orig):
    jg = []
    for i in range(len(df_orig.index) - 1) :
        jg.append(int((df_orig.index[i + 1] - df_orig.index[i]).seconds / 60))
    draw_pic(jg)
    return mode(jg)



def timeiter_fill(df_orig):
    #path_openfile_name = "E:\jupter\A-JRZ\data_t/0.csv"
    #df_orig = pd.read_csv(path_openfile_name, parse_dates=True, index_col=0,encoding='utf-8').head(1000)

    jg = []
    for i in range(len(df_orig.index) - 1) :
        jg.append(int((df_orig.index[i + 1] - df_orig.index[i]).seconds / 60))
    return mode(jg)



