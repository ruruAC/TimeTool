from downstream import writecsv
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
from numpy import *
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import math
import random   #test 1/5的测试集
import time
from timeinter import timeiter_fill,timeiter_fill_draw
from time_pre_fill import time_pre_csv
import streamlit as st
st.set_page_config(
    page_title="TimeToolv3 Web",
  page_icon="浙江大学.png",
    
     initial_sidebar_state="expanded", )
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib
matplotlib.use('Agg')
import datetime



def time_fill_csv(date_info,methord,min,pdtime,mising,test,pred):

    # 根据序列的内在属性以下是几种比较有效的填充方法：

    # 向后填充法
    # 线性插值法
    # 三次H插值法
    # 最近邻均值法

    df = pdtime.copy(deep=True)
    num_test=50


    ## 1. Actual -------------------------------
    if methord=='Actual' or methord=='All':

        # df_orig.plot(label='Actual', linewidth=3.0)
        # plt.legend(["Available Data"])
        #plt.show()
        if methord!='All':
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.head(1000).index, y=pdtime['value'].head(1000),
                marker_symbol="star"
            ))

            fig2.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                              showlegend=False)
            fig2.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时", step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            st.plotly_chart(fig2, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.index, y=pdtime['value'],
                marker_symbol="star"
            ))

            fig.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                              showlegend=False)
            fig.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时", step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )




    ## 2. Forward Fill --------------------------
    if methord == 'Forward_Fill' or methord == 'All':
        df_ffill = df.ffill()
        df['Forward_Fill'] = df_ffill['value']
       
        if methord!='All':
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.head(1000).index, y=pdtime["value"].head(1000),
                xperiodalignment="start"
            ))

            fig2.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.head(1000).index, y=df_ffill['value'].head(1000),
                marker_symbol="star"
            ))


            fig2.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                              showlegend=False)
            fig2.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时",step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            st.plotly_chart(fig2, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.index, y=pdtime["value"],
                xperiodalignment="start"
            ))

            fig.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.index, y=df_ffill['value'],
                marker_symbol="star"
            ))

            fig.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                              showlegend=False)
            fig.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时", step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )


## 3. Backward Fill -------------------------
    if methord == 'Backward_Fill' or methord == 'All':
        df_bfill = df.bfill()
        df['Backward_Fill'] = df_bfill['value']
        if methord!='All':
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.head(1000).index, y=pdtime["value"].head(1000),
                xperiodalignment="start"
            ))
            fig2.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.head(1000).index, y=df_bfill['value'].head(1000),
                marker_symbol="star"
            ))
            fig2.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                              showlegend=False)
            fig2.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时", step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.index,y=df_bfill['value'],
                marker_symbol="star"
            ))
            fig.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.index, y=pdtime["value"],
                xperiodalignment="start"
            ))
            fig.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                             showlegend = False)
            fig.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时", step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(fig2, use_container_width=True)


    df['rownum'] = np.arange(df.shape[0])
    df_nona = df.dropna(subset=['value'])
## 4. Linear Interpolation ------------------ 从一个NA的相邻两个真实点(相邻就是最近的两个点，如果是两边正好平均，否则就是一侧的两个最近点） 代入y=ax+b
    if methord == 'Linear_Interpolation' or methord == 'All':


        f = interp1d(df_nona['rownum'], df_nona['value'], fill_value="extrapolate")
        df['Linear_Interpolation'] = f(df['rownum'])
        if methord!='All':
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.head(1000).index, y=pdtime["value"].head(1000),
                xperiodalignment="start"
            ))

            fig2.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.head(1000).index, y=df['Linear_Interpolation'].head(1000),
                marker_symbol="star"
            ))

            fig2.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                              showlegend=False)
            fig2.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时", step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )


            fig = go.Figure()
            fig.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines",x=pdtime.index, y=pdtime["value"],
                xperiodalignment="start"
            ))

            fig.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.index, y=df['Linear_Interpolation'],
                marker_symbol="star"
            ))


            fig.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                              showlegend=False)
            fig.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时", step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(fig2, use_container_width=True)


## 5. Cubic Interpolation -------------------- 三次Hermite插值，采用H多项式，在斜率选择上 略微不同于三次样本插值
    if methord == 'Cubic_Interpolation' or methord == 'All':
        f2 = interp1d(df_nona['rownum'], df_nona['value'], kind='cubic', fill_value="extrapolate")
        df['Cubic_Interpolation'] = f2(df['rownum'])
        if methord!='All':
            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.head(1000).index, y=pdtime['value'].head(1000),
                xperiodalignment="start"
            ))
            fig2.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.head(1000).index, y=df['Cubic_Interpolation'].head(1000),
                marker_symbol="star"
            ))


            fig2.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                              showlegend=False)
            fig2.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时", step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.index, y=pdtime['value'],
                xperiodalignment="start"
            ))
            fig.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.index, y=df['Cubic_Interpolation'],
                marker_symbol="star"
            ))


            fig.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                              showlegend=False)
            fig.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时", step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(fig2, use_container_width=True)


    def knn_mean(ts, n) :
        out = np.copy(ts)
        for i, val in enumerate(ts) :
            if np.isnan(val) :
                n_by_2 = np.ceil(n / 2)
                lower = np.max([0, int(i - n_by_2)])
                upper = np.min([len(ts) + 1, int(i + n_by_2)])
                ts_near = np.concatenate([ts[lower :i], ts[i :upper]])
                out[i] = np.nanmean(pd.to_numeric(ts_near))
        return out

    if methord == 'knn_mean' or methord == 'All' :
        df['knn_mean'] = knn_mean(df.value.values, 8)
        if methord!='All':
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.head(1000).index, y=pdtime["value"].head(1000),
                xperiodalignment="start"
            ))
            fig2.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.head(1000).index, y=df['knn_mean'].head(1000),
                marker_symbol="star"
            ))




            fig2.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                              showlegend=False)
            fig2.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时", step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )


            fig = go.Figure()
            fig.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.index, y=pdtime["value"],
                xperiodalignment="start"
            ))

            fig.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.index, y=df['knn_mean'],
                marker_symbol="star"
            ))


            fig.update_layout(width=1000,
                              height=600,
                              xaxis_showgrid=False,
                              yaxis_showgrid=False,

                              showlegend=False)
            fig.update_xaxes(
                showgrid=True,
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1小时", step="hour", stepmode="backward"),
                        dict(count=1, label="1天", step="day", stepmode="backward"),
                        dict(count=1, label="1个月", step="month", stepmode="todate"),
                        dict(count=1, label="1年", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(fig2, use_container_width=True)


    #print("缺失值个数：", len(mising))
    if methord == 'All' :
        print("插值方法为：'Forward Fill', 'Backward Fill', 'Linear Interpolation', 'Cubic Interpolation','knn_mean'")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            name="Actual",
            mode="markers+lines", x=df.head(1000).index, y=df["value"].head(1000),
            xperiodalignment="start"
        ))

        fig2.add_trace(go.Scatter(
            name='Forward Fill',
            mode="markers+lines", x=df.head(1000).index, y=df['Forward_Fill'].head(1000),
            marker_symbol="star"
        ))
        fig2.add_trace(go.Scatter(
            name='Backward Fill',
            mode="markers+lines", x=df.head(1000).index, y=df['Backward_Fill'].head(1000),
            xperiodalignment="start"
        ))
        fig2.add_trace(go.Scatter(
            name='Linear Interpolation',
            mode="markers+lines", x=df.head(1000).index, y=df['Linear_Interpolation'].head(1000),
            xperiodalignment="start"
        ))
        fig2.add_trace(go.Scatter(
            name='Cubic Interpolation',
            mode="markers+lines", x=df.head(1000).index, y=df['Cubic_Interpolation'].head(1000),
            xperiodalignment="start"
        ))
        fig2.add_trace(go.Scatter(
            name='knn mean',
            mode="markers+lines", x=df.head(1000).index, y=df['knn_mean'].head(1000),
            xperiodalignment="start"
        ))

        fig2.update_layout(width=1000,
                          height=600,
                          xaxis_showgrid=False,
                          yaxis_showgrid=False,

                          showlegend=False)
        fig2.update_xaxes(
            showgrid=True,
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1小时", step="hour", stepmode="backward"),
                    dict(count=1, label="1天", step="day", stepmode="backward"),
                    dict(count=1, label="1个月", step="month", stepmode="todate"),
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Actual",
            mode="markers+lines", x=df.index, y=df["value"],
            xperiodalignment="start"
        ))

        fig.add_trace(go.Scatter(
            name='Forward Fill',
            mode="markers+lines", x=df.index, y=df['Forward_Fill'],
            marker_symbol="star"
        ))
        fig.add_trace(go.Scatter(
            name='Backward Fill',
            mode="markers+lines", x=df.index, y=df['Backward_Fill'],
            xperiodalignment="start"
        ))
        fig.add_trace(go.Scatter(
            name='Linear Interpolation',
            mode="markers+lines", x=df.index, y=df['Linear_Interpolation'],
            xperiodalignment="start"
        ))
        fig.add_trace(go.Scatter(
            name='Cubic Interpolation',
            mode="markers+lines", x=df.index, y=df['Cubic_Interpolation'],
            xperiodalignment="start"
        ))
        fig.add_trace(go.Scatter(
            name='knn mean',
            mode="markers+lines", x=df.index, y=df['knn_mean'],
            xperiodalignment="start"
        ))



        fig.update_layout(width=1000,
                          height=600,
                          xaxis_showgrid=False,
                          yaxis_showgrid=False,

                          showlegend=False)
        fig.update_xaxes(
            showgrid=True,
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1小时", step="hour", stepmode="backward"),
                    dict(count=1, label="1天", step="day", stepmode="backward"),
                    dict(count=1, label="1个月", step="month", stepmode="todate"),
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        st.plotly_chart(fig2, use_container_width=True)



    ## 2. Forward Fill --------------------------
    def test_ff() :
        val['rownum'] = np.arange(val.shape[0])
        df_ffill = val.ffill()
        val['Forward_Fill']=df_ffill['value']
        error = np.round(mean_squared_error(test['value'], df_ffill['value']) * num_test / (int(num_test / 5)), 8)

        return error

    ## 3. Backward Fill -------------------------
    def test_bf() :
        val['rownum'] = np.arange(val.shape[0])
        df_bfill = val.ffill()
        val['Backward_Fill'] = df_bfill['value']
        error = np.round(mean_squared_error(test['value'], df_bfill['value']) * num_test / (int(num_test / 5)), 8)

        return error

    ## 4. Linear Interpolation ------------------
    def test_li() :
        val['rownum'] = np.arange(val.shape[0])
        df_nona = val.dropna(subset=['value'])
        f = interp1d(df_nona['rownum'], df_nona['value'], fill_value="extrapolate")
        val['linear_fill'] = f(val['rownum'])
        error = np.round(mean_squared_error(test['value'], val['linear_fill']) * num_test / (int(num_test / 5)), 8)

        return error

    ## 5. Cubic Interpolation --------------------
    def test_Ci() :
        val['rownum'] = np.arange(val.shape[0])
        df_nona = val.dropna(subset=['value'])
        f2 = interp1d(df_nona['rownum'], df_nona['value'], kind='cubic', fill_value="extrapolate")
        val['cubic_fill'] = f2(val['rownum'])
        error = np.round(mean_squared_error(test['value'], val['cubic_fill']) * num_test / (int(num_test / 5)), 8)
        # pd.merge(test, val['cubic_fill'], left_index=True, right_index=True).plot(
        #     title="Cubic Fill (MSE: " + str(error) + ")", label='Cubic Fill', style=".-", linewidth=3.0)
        # plt.show()
        return error

    ## 6. Mean of 'n' Nearest Past Neighbors ------


    def test_knn() :
        val['rownum'] = np.arange(val.shape[0])

        val['knn_mean'] = knn_mean(val.value.values, 8)
        error = np.round(mean_squared_error(test['value'], val['knn_mean']) * num_test / (int(num_test / 5)), 8)

        return error

    def draw(test, val) :
        pd.merge(test, val, left_index=True, right_index=True).plot(label='Actual', linewidth=3.0,linestyle='--', marker = 'D', ms = 5,mfc = 'r')
        plt.grid()
        plt.legend(["Missing Data", "Available Data"])
        plt.show()

    if methord =='test-50' or methord=='All':
        ff = []
        bf = []
        li = []
        ci = []
        knn = []
        test_error = [ff, bf, li, ci, knn]
        if methord == 'test-50':
            fig=''

        for t in range(0, 5) :
            global val
            val = test.copy(deep=True)
            alist = random.sample(range(1, num_test), int(num_test / 5))  # random.sample()生成不相同的随机数
            val.value[alist] = np.NaN
            ## 1. Actual -------------------------------
            #draw(test, val)
            ff.append(test_ff())
            bf.append(test_bf())
            li.append(test_li())
            ci.append(test_Ci())
            knn.append(test_knn())
            val.drop(['rownum'], axis=1).plot(title=" Test "+ str(t+1),style=".-", linewidth=3.0)

        # 求每个列表平均值 返回最大那个，可以用遍历加索引，然后得到最优的方法
        # 然后用最优的方法去插值所有
        # 连续缺失值应该有一个标记 后者直接knn nan状态就是异常的
        # 也可以全部方法都用 然后再可以验证

        st.subheader("每种方法均方差比较：")
        st.text(" 前向插值（Forward Fill） 均方差: %.8f" % mean(ff))
        st.text(" 后向插值（Backward Fill） 均方差:%.8f" % mean(bf))
        st.text(" 线性插值（Linear Interpolation） 均方差:%.8f" % mean(li))
        st.text(" 三次Hermite插值（Cubic Interpolation） 均方差:%.8f" % mean(ci))
        st.text(" 最近邻插值（Mean of 'n' Nearest Past Neighbors） 均方差:%.8f" % mean(knn))
        df.loc[df.index.values[0],'info'] =[""]
        
        df.rename(columns={'info':"“Value”异常个数：%d；" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                            +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'+'每种方法均方差：'+'\n'+"前向插值（Forward Fill） 均方差: %.8f；" % mean(ff)+'\n'+"后向插值（Backward Fill）均方差:%.8f；" % mean(bf)+'\n'+"线性插值（Linear Interpolation）均方差:%.8f；" % mean(li)+'\n'+"三次Hermite插值（Cubic Interpolation）均方差:%.8f；" % mean(ci)+\
                                            '\n'+"最近邻插值（knn_mean）均方差:%.8f。" % mean(knn)},inplace=True)


    elif methord =='Forward_Fill':
        ff = []
        for t in range(0, 5) :

            val = test.copy(deep=True)
            alist = random.sample(range(1, num_test), int(num_test / 5))  # random.sample()生成不相同的随机数
            val.value[alist] = np.NaN
            ## 1. Actual -------------------------------
            #draw(test, val)
            ff.append(test_ff())
            #val.drop(['rownum'], axis=1).plot(title=" Test "+ str(t+1),style=".-", linewidth=3.0)

        
        st.sidebar.write("前向插值（Forward Fill） 均方差: %.8f" % mean(ff))
        
        df.loc[df.index.values[0],'info']= [""]
                                     
        df.rename(columns={'info':"“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                            +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'\
                                            +"前向插值（Forward Fill） 均方差: %.8f。" % mean(ff)},inplace=True)
        df4.loc[df.index.values[0],'info']= ["“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                            +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'\
                                            +"前向插值（Forward Fill） 均方差: %.8f。" % mean(ff)]
                                     
      

    elif methord == 'Backward_Fill':
        bf = []
        for t in range(0, 5) :

            val = test.copy(deep=True)
            alist = random.sample(range(1, num_test), int(num_test / 5))  # random.sample()生成不相同的随机数
            val.value[alist] = np.NaN
            ## 1. Actual -------------------------------
            # draw(test, val)
            bf.append(test_bf())
            #val.drop(['rownum'], axis=1).plot(title=" Test " + str(t + 1), style=".-", linewidth=3.0)
        st.sidebar.write("后向插值（Backward Fill）均方差: %.8f。" % mean(bf))

        df.loc[df.index.values[0],'info'] = [""]
        df.rename(columns={'info':"“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                    +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'\
                                    +"后向插值（Backward Fill）均方差: %.8f。" % mean(bf)},inplace=True)
        df4.loc[df.index.values[0],'info']=["“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                    +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'\
                                    +"后向插值（Backward Fill）均方差: %.8f。" % mean(bf)]

    elif methord =='Linear_Interpolation':
        li = []
        for t in range(0, 5) :

            val = test.copy(deep=True)
            alist = random.sample(range(1, num_test), int(num_test / 5))  # random.sample()生成不相同的随机数
            val.value[alist] = np.NaN
            ## 1. Actual -------------------------------
            # draw(test, val)
            li.append(test_li())
            #val.drop(['rownum'], axis=1).plot(title=" Test " + str(t + 1), style=".-", linewidth=3.0)

        st.sidebar.write("线性插值（Linear Interpolation）均方差: %.8f。" % mean(li))
        df.loc[df.index.values[0],'info']=[""]
        df.rename(columns={'info':"“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                    +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'\
                                    +"线性插值（Linear Interpolation） 均方差: %.8f。" % mean(li)},inplace=True)
        df4.loc[df.index.values[0],'info']=["“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                    +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'\
                                    +"线性插值（Linear Interpolation） 均方差: %.8f。" % mean(li)]


    elif methord =='Cubic_Interpolation':
        ci = []
        for t in range(0, 5) :

            val = test.copy(deep=True)
            alist = random.sample(range(1, num_test), int(num_test / 5))  # random.sample()生成不相同的随机数
            val.value[alist] = np.NaN
            ## 1. Actual -------------------------------
            # draw(test, val)
            ci.append(test_Ci())
            #val.drop(['rownum'], axis=1).plot(title=" Test " + str(t + 1), style=".-", linewidth=3.0)

        st.sidebar.write("三次Hermite插值（Cubic Interpolation）均方差: %.8f。" % mean(ci))
        df.loc[df.index.values[0],'info'] =[""]
        df.rename(columns={'info':"“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                    +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'\
                                    +"三次Hermite插值（Cubic Interpolation）均方差: %.8f。" % mean(ci)},inplace=True)
        df4.loc[df.index.values[0],'info']=["“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                    +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'\
                                    +"三次Hermite插值（Cubic Interpolation）均方差: %.8f。" % mean(ci)]


    elif methord == 'knn_mean':
        knn = []
        for t in range(0, 5) :

            val = test.copy(deep=True)
            alist = random.sample(range(1, num_test), int(num_test / 5))  # random.sample()生成不相同的随机数
            val.value[alist] = np.NaN
            ## 1. Actual -------------------------------
            # draw(test, val)
            knn.append(test_knn())
            #val.drop(['rownum'], axis=1).plot(title=" Test " + str(t + 1), style=".-", linewidth=3.0)

        st.sidebar.write("最近邻插值（knn_mean）均方差:%.8f。" % mean(knn))
        df.loc[df.index.values[0],'info'] = [""]
        df.rename(columns={'info':"“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                    +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'\
                                    +"最近邻插值（knn_mean）均方差:%.8f。" % mean(knn)},inplace=True)
        df4.loc[df.index.values[0],'info']=["“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                    +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'\
                                    +"最近邻插值（knn_mean）均方差:%.8f。" % mean(knn)]


    else:
        df.loc[df.index.values[0],'info'] = ["“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                    +'\n'+"有效值总数为:%d；"%date_info[4]+'\n']
       #如果是插值
        df.rename(columns={'info':"“Value”异常个数：%d:" %date_info[1]+'\n'+"“Date”异常个数：%d；" %date_info[3]+'\n'+"“Date”重复个数：%d；"%date_info[2]+'\n'+"缺失值总数为:%d；"%len(mising) \
                                    +'\n'+"有效值总数为:%d；"%date_info[4]+'\n'},inplace=True)

   
    if methord !='knn_mean' and methord!='test-50' and methord!='All' and pred==1: #预测,前30要填充
        df['knn_mean'] = knn_mean(df.value.values, 8)
        row_indexer2=df['knn_mean'].isnull()
        row_indexer=row_indexer2
        for i in range(len(row_indexer2)):
            
            if i<=30 and row_indexer[i]=='True' :
                row_indexer[i]='False'
                
        df.loc[row_indexer] = np.NaN
        df.loc[row_indexer2,'missing']='连续缺失'
        df=df.drop(['knn_mean'], axis=1)
        
    elif  methord !='knn_mean' and methord!='test-50' and methord!='All':
        df['knn_mean'] = knn_mean(df.value.values, 8)
        row_indexer2=df['knn_mean'].isnull()
             
        df.loc[row_indexer2] = np.NaN
        df.loc[row_indexer2,'missing']='连续缺失'

        df=df.drop(['knn_mean'], axis=1)
    elif methord != 'test-50':
        row_indexer2=df['knn_mean'].isnull()
        row_indexer=[]
        for r in row_indexer2:
            if r>30:
                row_indexer.append(r)
                
        df.loc[row_indexer] = np.NaN
        df.loc[row_indexer2,'missing']='连续缺失'

    return df,fig


def is_number(s) :
    try :
        float(s)
        return True
    except ValueError :
        pass

    try :
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError) :
        pass

    return False


@st.cache(suppress_st_warning=True)
def get_data(df_orig):
    yichang = 0
    date_info = []
    date_info.append(len(df_orig))

    for i in range(len(df_orig)) :  # 去除value异常值
        if is_number(df_orig.values[i][0]) == False :
            yichang = yichang + 1
            df_orig.values[i][0] = np.NaN
            print(df_orig.values[i][0])
        else :

            df_orig.value[i] = float(df_orig.value[i])

    print('删除异常值个数:', yichang)

    date_info.append(yichang)

    fl = len(df_orig)
    df_orig = df_orig[~df_orig.index.duplicated()]  # 删除date索引的重复值，重复日期，保留第一个
    print('删除重复值个数:', fl - len(df_orig))

    date_info.append(fl - len(df_orig))
    fl = len(df_orig)

    if isinstance(df_orig.index, pd.DatetimeIndex) :  # 全是数字
        print("全是数字")
        df_orig.sort_index(inplace=True)
        date_info.append(0)

    elif isinstance(df_orig.index, pd.Index) :  # 有字符，需要找到是字符的，不符合日期要求的数字也会变成字符
        print("有字符异常")
        print(df_orig.index)
        import time
        for i in range(len(df_orig) - 1) :
            if i >= len(df_orig) :
                break

            try :
                time.strptime(df_orig.index[i], '%Y-%m-%d %H:%M:%S.%f')
            # except Exception as e :
            except :
                try :
                    time.strptime(df_orig.index[i], '%Y-%m-%d %H:%M:%S')
                except :
                    try :
                        time.strptime(df_orig.index[i], '%Y/%m/%d %H:%M:%S')
                    except :
                        try :
                            time.strptime(df_orig.index[i], '%Y-%m-%d %H:%M')

                        except :
                            try :
                                time.strptime(df_orig.index[i], '%Y/%m/%d %H:%M')
                            except :

                                df_orig.drop(index=df_orig.index[i], inplace=True)  # 删除时注意索引！
                            else :
                                if datetime.datetime.strptime(df_orig.index[i],
                                                              '%Y/%m/%d %H:%M') > datetime.datetime.now() :
                                    df_orig.drop(index=df_orig.index[i], inplace=True)  # 删除时注意索引！
                        else :
                            if datetime.datetime.strptime(df_orig.index[i],
                                                          '%Y-%m-%d %H:%M') > datetime.datetime.now() :
                                df_orig.drop(index=df_orig.index[i], inplace=True)  # 删除时注意索引！
                    else :
                        if datetime.datetime.strptime(df_orig.index[i], '%Y-%m-%d %H:%M') > datetime.datetime.now() :
                            df_orig.drop(index=df_orig.index[i], inplace=True)  # 删除时注意索引！
                else :
                    if datetime.datetime.strptime(df_orig.index[i], '%Y-%m-%d %H:%M:%S') > datetime.datetime.now() :
                        df_orig.drop(index=df_orig.index[i], inplace=True)
            else :
                if datetime.datetime.strptime(df_orig.index[i], '%Y-%m-%d %H:%M:%S.%f') > datetime.datetime.now() :
                    df_orig.drop(index=df_orig.index[i], inplace=True)

            # else:
            #     df_orig.index[i]=pd.DatetimeIndex(df_orig.index[i])
            #     print("越界",df_orig.index[i],datetime.datetime.now())
            #     if df_orig.index[i] <datetime.datetime.now() :
            #         df_orig.drop(index=df_orig.index[i], inplace=True)  # 删除时注意索引！

        df_orig.index = pd.DatetimeIndex(df_orig.index)  # 如果开头结尾有重复日期就不能转换
        df_orig.sort_index(inplace=True)
        print("删去异常值个数：", fl - len(df_orig))
        date_info.append(fl - len(df_orig))
    date_info.append(len(df_orig))

    return df_orig, date_info






@st.cache(suppress_st_warning=True)
def get_data_pd(df_orig):

    global mising, test, time, pdtime

    bar = st.progress(0)
    placeholder = st.empty()
    placeholder.text("%d %%" % 0)

    df_orig.columns=['value']


    if 1 :
        jindu = 0
        #df_orig= pd.read_csv(uploaded_file,parse_dates=True, index_col=0,encoding='utf-8')
        min = timeiter_fill(df_orig)
        print(min)
        min = str(min) + 'T'
        time2 = pd.date_range(df_orig.index[0], df_orig.index[len(df_orig.index) - 1], freq=min)  # T/MIN：每分
        pdtime = pd.DataFrame(columns=('date', 'value', 'missing'))

        t = 0
        mising = []
        test = pd.DataFrame(columns=('date', 'value'))
        num_test = 50
        flag = 0
        nn=0

        for date in time2 :
            date = date.strftime("%Y/%m/%d %H:%M:%S")
            # if date==df_orig.index[t].strftime("%Y/%m/%d %H:%M:%S"):
            #     print(date,df_orig.index[t].strftime("%Y/%m/%d %H:%M:%S"),"相等")
            # else:
            #     print(date, df_orig.index[t].strftime("%Y/%m/%d %H:%M:%S"), "不相等")

            if 1:

                if math.isnan(df_orig.value[t]) :
                    if flag != 1 and len(test) != 0 :
                        test = test.iloc[0 :0].copy()
                    pdtime = pdtime.append(
                        pd.DataFrame({'date' : date, 'value' : [np.NaN], 'missing' : ['缺失']}))  # 必须赋值！不然还是空
                    t = t + 1
                    mising.append(date)

                elif date == df_orig.index[t].strftime("%Y/%m/%d %H:%M:%S") :
                    if len(test) < num_test :
                        test = test.append(pd.DataFrame({'date' : date, 'value' : [df_orig.value[t]]}))
                    if len(test) == num_test :
                        flag = 1
                    pdtime = pdtime.append(
                        pd.DataFrame({'date' : date, 'value' : [df_orig.value[t]], 'missing' : ['未缺失']}))
                    t = t + 1

                elif date > df_orig.index[t].strftime("%Y/%m/%d %H:%M:%S") :
                    if flag != 1 and len(test) != 0 :
                        test = test.iloc[0 :0].copy()
                    pdtime = pdtime.append(pd.DataFrame({'date' : date, 'value' : [np.NaN], 'missing' : ['缺失']}))
                    mising.append(date)
                    t = t + 1
                else :
                    if flag != 1 and len(test) != 0 :
                        test = test.iloc[0 :0].copy()
                    pdtime = pdtime.append(pd.DataFrame({'date' : date, 'value' : [np.NaN], 'missing' : ['缺失']}))
                    mising.append(date)

            else:
                if flag != 1 and len(test) != 0 :
                    test = test.iloc[0 :0].copy()
                pdtime = pdtime.append(pd.DataFrame({'date' : date, 'value' : [np.NaN], 'missing' : ['缺失']}))
                mising.append(date)

            nn = nn + 1
            if nn == int(len(time2) * 1 / 20) :
                if jindu==100:
                    break
                jindu = jindu + 5
                bar.progress(jindu)
                nn = 0
                placeholder.text("%d %%"%jindu)

        st.success('Preprocessing Done!')

        from functools import partial
        to_datetime_fmt = partial(pd.to_datetime, format="%Y/%m/%d %H:%M:%S")
        pdtime['date'] = pdtime['date'].apply(to_datetime_fmt)
        pdtime = pdtime.set_index(['date'])
        test['date'] = test['date'].apply(to_datetime_fmt)
        test = test.set_index(['date'])

    return pdtime,mising,test
@st.cache(suppress_st_warning=True)

def get_data_pd2(df_orig):

    global mising, test, time, pdtime

    bar = st.progress(0)
    placeholder = st.empty()
    placeholder.text("%d %%" % 0)

    df_orig.columns=['value']


    if 1 :
        jindu = 0
        #df_orig= pd.read_csv(uploaded_file,parse_dates=True, index_col=0,encoding='utf-8')
        min = timeiter_fill(df_orig)
        print(min)
        min = str(min) + 'T'
        time2 = pd.date_range(df_orig.index[0], df_orig.index[len(df_orig.index) - 1], freq=min)  # T/MIN：每分
        pdtime = pd.DataFrame(columns=('date', 'value', 'predict'))

        t = 0
        mising = []
        test = pd.DataFrame(columns=('date', 'value'))
        num_test = 50
        flag = 0
        nn=0

        for date in time2 :
            date = date.strftime("%Y/%m/%d %H:%M:%S")
            # if date==df_orig.index[t].strftime("%Y/%m/%d %H:%M:%S"):
            #     print(date,df_orig.index[t].strftime("%Y/%m/%d %H:%M:%S"),"相等")
            # else:
            #     print(date, df_orig.index[t].strftime("%Y/%m/%d %H:%M:%S"), "不相等")

            if 1:

                if math.isnan(df_orig.value[t]) :
                    if flag != 1 and len(test) != 0 :
                        test = test.iloc[0 :0].copy()
                    pdtime = pdtime.append(
                        pd.DataFrame({'date' : date, 'value' : [np.NaN], 'predict' : ['预测']}))  # 必须赋值！不然还是空
                    t = t + 1
                    mising.append(date)

                elif date == df_orig.index[t].strftime("%Y/%m/%d %H:%M:%S") :
                    if len(test) < num_test :
                        test = test.append(pd.DataFrame({'date' : date, 'value' : [df_orig.value[t]]}))
                    if len(test) == num_test :
                        flag = 1
                    pdtime = pdtime.append(
                        pd.DataFrame({'date' : date, 'value' : [df_orig.value[t]], 'predict' : ['未缺失']}))
                    t = t + 1

                elif date > df_orig.index[t].strftime("%Y/%m/%d %H:%M:%S") :
                    if flag != 1 and len(test) != 0 :
                        test = test.iloc[0 :0].copy()
                    pdtime = pdtime.append(pd.DataFrame({'date' : date, 'value' : [np.NaN], 'predict' : ['预测']}))
                    mising.append(date)
                    t = t + 1
                else :
                    if flag != 1 and len(test) != 0 :
                        test = test.iloc[0 :0].copy()
                    pdtime = pdtime.append(pd.DataFrame({'date' : date, 'value' : [np.NaN], 'predict' : ['预测']}))
                    mising.append(date)

            else:
                if flag != 1 and len(test) != 0 :
                    test = test.iloc[0 :0].copy()
                pdtime = pdtime.append(pd.DataFrame({'date' : date, 'value' : [np.NaN], 'predict' : ['预测']}))
                mising.append(date)

            nn = nn + 1
            if nn == int(len(time2) * 1 / 20) :
                if jindu==100:
                    break
                jindu = jindu + 5
                bar.progress(jindu)
                nn = 0
                placeholder.text("%d %%"%jindu)

        st.success('Preprocessing Done!')

        from functools import partial
        to_datetime_fmt = partial(pd.to_datetime, format="%Y/%m/%d %H:%M:%S")
        pdtime['date'] = pdtime['date'].apply(to_datetime_fmt)
        pdtime = pdtime.set_index(['date'])
        test['date'] = test['date'].apply(to_datetime_fmt)
        test = test.set_index(['date'])

    return pdtime,mising,test

if __name__ == '__main__' :
    import cartoon_html

    cartoon_html.cartoon_html()

    # st.title("TimeToolv1")
    st.caption("Updated: Sep 29, 2021")

#改了try
    if 1:



        st.subheader('**选择时序数据处理功能** :wave:')


        #
        funn = st.selectbox(
            '选择如下：',
            ('插值', '预测','长期预测'))
        st.write('您选择了:', funn)
        st.subheader('**Step 1**: 上传文件 :wave:')

        uploaded_file = st.file_uploader('', type=['csv', 'xlsx'], key=None)
        st.sidebar.title("操作面板")
        df_orig = pd.DataFrame()
        global mising, test, time, pdtime
        df_orig2=pd.DataFrame()

        if uploaded_file is not None :

            # if uploaded_file.name[-4 :] == 'xlsx':
            if uploaded_file.name[-4 :] == 'xlsx' :

                df_orig = pd.read_excel(uploaded_file, parse_dates=True, index_col=0)

            else :
                try :
                    df_orig = pd.read_csv(uploaded_file, parse_dates=True, index_col=0, encoding='gb18030')
                except :
                    df_orig = pd.read_csv(uploaded_file, parse_dates=True, index_col=0, encoding='utf-8')

            ls = list(df_orig)
            if funn!='长期预测':

                st.subheader("**您选择对哪一列时序数据进行分析？**:wave:")
                value_col = st.sidebar.selectbox(
                    '时序数据列选择如下：',
                    (ls))
                st.write("您选择的数据对象是：", value_col)
                df_orig2 = df_orig.copy(deep=True)

                for col in list(df_orig) :
                    if col != value_col :
                        df_orig2.drop(col, axis=1, inplace=True)

                df_orig2.columns = ['value']


        # file ="E:\jupter/test\TimeToolv1\example/aa温度-data-2021-08-13 16_32_39.xlsx"

        #print(df_orig2)
        if funn!='长期预测':
            df1, date_info = get_data(df_orig2)  # df1去除异常之后的数据
            df_chu = df1.copy(deep=True)
            st.sidebar.write(" ")
            st.sidebar.write("**数据预处理结果**")
            st.sidebar.write("初始数据个数：", date_info[0])  # 初始
            st.sidebar.write("“Value”异常个数：", date_info[1])  # 异常
            st.sidebar.write("“Date”重复个数：", date_info[2])  # 异常
            st.sidebar.write("“Date”异常个数：", date_info[3])  # 异常
            st.sidebar.write("有效数据总数：", date_info[4])  # 处理后数据有效值

        if funn=="长期预测":

            st.subheader("您选择对哪一列时序数据进行输入？:wave:")
            value_col = st.sidebar.multiselect(
                '时序数据列选择如下：',
                (ls))
            st.write("您选择的输入对象是：", value_col)
            st.subheader("您选择对哪一列时序数据进行输出？:wave:")
            value_col2 = st.sidebar.selectbox(
                '时序数据列选择如下：',
                (ls))

            st.write("您选择的输出对象是：", value_col2)
            df_orig2 = df_orig.copy(deep=True)
            value_col.append(value_col2)

            for col in list(df_orig) :

                if col not in value_col :
                    df_orig2.drop(col, axis=1, inplace=True)


            df1=df_orig2
            df1.insert(0, value_col2, df1.pop(value_col2))
            st.write(df1)
        st.subheader('**Step 2**: 选择时序范围 :wave:')

        (start_time, end_time) = st.select_slider("时序范围：",
                                                  options=df1.index,
                                                  value=((df1.index)[0], (df1.index)[len(df1) - 1],),
                                                  help="拖拽选择时序范围",
                                                  )

        st.write("开始时间:", start_time)
        st.write("结束时间:", end_time)

        # setting index as date

        df1 = df1[start_time :end_time]
        global df4
        df4=pd.DataFrame()
        # st.write("View the processed data (first 200 lines):")
        st.dataframe(df1.head(200))
        with st.expander("查看完整数据表格：") :
            st.write(df1)
        # ls=list(df1)
        # st.subheader("**您选择对哪一列时序数据进行分析？**:wave:")
        # value_col = st.sidebar.selectbox(
        #     '选择如下：',
        #     (ls))
        # st.write("您选择的数据对象是：",value_col)


        if 1:

            if funn == '插值' :  # 上传文件缓存前不能加按钮


                st.subheader('**Step 3**: 您选择哪种插值方法？ :wave:')
                # st.sidebar.image("浙大.jpg")

                methord = st.sidebar.selectbox(
                    '插值算法选择如下：',
                    ('Actual', 'Forward_Fill', 'Backward_Fill', 'Linear_Interpolation', 'Cubic_Interpolation',
                     'knn_mean',
                     'test-50', 'All'))
                st.write('您选择了：', methord)
                if methord == 'Actual' :
                    st.sidebar.write("真实值：输出清洗过异常数据和重复数据，同时按照最大频次时间间隔切片处理的真实数据结果。")
                elif methord == 'Forward_Fill' :
                    st.sidebar.write("前向插值：当前缺失值按照前一个已知值进行填充。如果缺失值前面没有真实值，则插值后仍为空值。")
                elif methord == 'Backward_Fill' :
                    st.sidebar.write("后向插值：当前缺失值按照后一个已知值进行填充。如果缺失值后面没有真实值，则插值后仍为空值。")
                elif methord == 'Linear_Interpolation' :
                    st.sidebar.write("线性插值：通过连接相邻两个已知值的直线，以对在这两个已知值之间的缺失值进行填充。")
                elif methord == 'Cubic_Interpolation' :
                    st.sidebar.write(
                        "三次Hermite插值：通过在相邻两个已知值之间采用不超过三次的多项式函数将其连接，并且要求该函数二阶以下均可导的要求下，构建三次Hermite多项式，对当前缺失值进行插值。")
                elif methord == 'knn_mean' :
                    st.sidebar.write("最近邻插值：当前缺失值按照前4个时刻和后4个时刻已知值的平均值进行填充。如果前4个时刻和后4个时刻都为空值，则插值后仍为空值。")
                elif methord == 'test-50' :
                    st.sidebar.write("交叉验证：基于没有缺失的连续时间序列数据，对算法进行验证，输出每种插值算法的均方误差MSE。")
                elif methord == 'All' :
                    st.sidebar.write("所有插值算法：输出以上所有插值算法的插值结果。")
                min = timeiter_fill_draw(df1)
                agree = st.sidebar.checkbox("下载", help="选择是否下载结果文件")
                agree2 = st.sidebar.button("开始", help="运行程序")


                global fig


                if agree and agree2 :
                    # for col in list(df1) :
                    #     if col != value_col :
                    #         df1 = df1.drop([col], axis=1)
                    pdtime, mising, test = get_data_pd(df1)  # pdtime是标记过每个间隔的数据

                    st.sidebar.write("**输出信息:**")
                    st.sidebar.write("缺失值个数：", len(mising))  # 处理后数据有效值
                    st.sidebar.write("时间间隔: ", min, "  Min ")
                    from downstream import writecsv

                    with st.spinner('Wait for it...') :
                        bar = st.progress(0)
                        placeholder = st.empty()
                        placeholder.text("Load the histogram...")

                        time.sleep(0.1)
                        bar.progress(30)
                        placeholder.text("Load the time series figure...")

                        df2,fig= time_fill_csv(date_info,methord, min, pdtime, mising, test,0)
                        time.sleep(0.1)
                        bar.progress(50)
                        with st.expander("查看完整图像 (注意:这可能会花费一些时间...)") :
                            if fig :
                                st.plotly_chart(fig, use_container_width=True)
                        placeholder.text("Load the time series form...")
                        st.subheader('**Step 4**: 输出文件 :wave:')
                        st.dataframe(df2.drop(['rownum'], axis=1).head(200))
                        bar.progress(70)
                        placeholder.text("Load the download url...")
                        name = methord + ".csv"
                        with st.expander("See  the entire output chart ") :
                            st.dataframe(df2.drop(['rownum'], axis=1))
                        st.subheader('**Step 5**: 文件下载链接 :wave:')

                        time1 = datetime.datetime.now()
                        now_time = datetime.datetime.strftime(time1, '%Y%m%d%H%M%S')
                        name = now_time + methord + ".csv"
                        kk = writecsv(df2.drop(['rownum'], axis=1), name)
                        st.write('http://120.26.89.97:8501/downloads/' + name)
                        st.sidebar.write('http://120.26.89.97:8501/downloads/' + name)
                        st.sidebar.caption("生成文件的下载链接")
                        bar.progress(100)
                        placeholder.text('Finish...')
                    st.success("Done !")

                elif agree2 :
                    # for col in list(df1) :
                    #     if col != value_col :
                    #         df1 = df1.drop([col], axis=1)
                    pdtime, mising, test = get_data_pd(df1)  # pdtime是标记过每个间隔的数据
                    st.sidebar.write("**输出信息：**")
                    st.sidebar.write("缺失值个数：", len(mising))  # 处理后数据有效值
                    st.sidebar.write("时间间隔: ", min, "  Min ")
                    with st.spinner('Wait for it...') :
                        bar = st.progress(0)
                        placeholder = st.empty()
                        placeholder.text("Load the histogram...")

                        time.sleep(0.1)
                        bar.progress(30)
                        placeholder.text("Load the time series figure...")

                        df2,fig= time_fill_csv(date_info,methord, min, pdtime, mising, test,0)
                        time.sleep(0.1)
                        bar.progress(50)
                        with st.expander("查看完整图像 (注意:这可能会花费一些时间...)") :
                            if fig :
                                st.plotly_chart(fig, use_container_width=True)
                        placeholder.text("Load the time series form...")
                        st.subheader('**Step 4**: 输出文件 :wave:')
                        st.write(df2.drop(['rownum'], axis=1).head(200))
                        bar.progress(70)
                        bar.progress(100)
                        placeholder.text('Finish...')
                    st.success("Done !")
                    st.sidebar.caption("Made by ruru@zju")
            if funn == '预测' :  # 上传文件缓存前不能加按钮
                st.subheader('**Step 3-1**: 您选择哪种插值方法？ :wave:')
                
                # st.sidebar.image("浙大.jpg")

                methord1 = st.sidebar.selectbox(
                    '插值算法选择如下：',
                    ( 'Forward_Fill', 'Backward_Fill', 'Linear_Interpolation', 'Cubic_Interpolation',
                     'knn_mean'
                     ))
                st.write('您选择了：', methord1)
                if methord1 == 'Actual' :
                    st.sidebar.write("真实值：输出清洗过异常数据和重复数据，同时按照最大频次时间间隔切片处理的真实数据结果。")
                elif methord1 == 'Forward_Fill' :
                    st.sidebar.write("前向插值：当前缺失值按照前一个已知值进行填充。如果缺失值前面没有真实值，则插值后仍未空值。")
                elif methord1 == 'Backward_Fill' :
                    st.sidebar.write("后向插值：当前缺失值按照后一个已知值进行填充。如果缺失值后面没有真实值，则插值后仍未空值。")
                elif methord1 == 'Linear_Interpolation' :
                    st.sidebar.write("线性插值：通过连接相邻两个已知值的直线，以对在这两个已知值之间的缺失值进行填充。")
                elif methord1 == 'Cubic_Interpolation' :
                    st.sidebar.write(
                        "三次Hermite插值：通过在相邻两个已知值之间采用不超过三次的多项式函数将其连接，并且要求该函数二阶以下均可导的要求下，构建三次Hermite多项式，对当前缺失值进行插值。")
                elif methord1 == 'knn_mean' :
                    st.sidebar.write("最近邻插值：当前缺失值按照前4个时刻和后4个时刻已知值的平均值进行填充。如果前4个时刻和后4个时刻都为空值，则插值后仍未空值。")
                elif methord1 == 'test-50' :
                    st.sidebar.write("交叉验证：基于没有缺失的连续时间序列数据，对算法进行验证，输出每种插值算法的均方误差MSE。")
                elif methord1 == 'All' :
                    st.sidebar.write("所有插值算法：输出以上所有插值算法的插值结果。")
                min = timeiter_fill_draw(df1)



                st.subheader('**Step 3-2**: 您选择哪种预测方法？ :wave:')
                # st.sidebar.image("浙大.jpg")

                methord2 = st.sidebar.selectbox(
                    '预测算法选择如下：',
                    ('Actual','Moving_Average', 'Linear_Regression', 'K_Nearest_Neighbours', 'ARIMA','LSTM','All'))
                st.write('您选择了：', methord2)
                if methord2=='Actual' :
                    st.sidebar.write("真实值：输出清洗过异常数据和重复数据，同时按照最大频次时间间隔切片处理的真实数据结果。")
                elif methord2 == 'Moving_Average' :
                    st.sidebar.write("移动平均：移动平均是用来衡量当前趋势的方向。移动平均和一般意义下的平均概念是一致的，都是通过计算过去数据的平均值得到的数学结果。如果缺失值前面没有真实值，则预测后仍为空值。")
                elif methord2 == 'Linear_Regression':
                    st.sidebar.write("线性回归：线性回归模型返回一个方程，该方程确定时间的相关变量，例如：年，月，日，星期，时间戳等，和因变量之间的关系。")
                elif methord2 == 'K_Nearest_Neighbours' :
                    st.sidebar.write("最近邻：通过衡量时间相关变量，例如：年，月，日，星期，时间戳等与因变量的关系，对因变量进行聚类，从而预测因变量的值。")
                elif methord2 == 'ARIMA' :
                    st.sidebar.write(
                        "ARIMA：ARIMA是一种非常流行的时间序列预测统计方法。ARIMA模型将过去的值考虑在内，以预测将来的值。如果缺失值前的训练数据少于一定个数，则不能对缺失值进行预测。")
                elif methord2 == 'LSTM' :
                    st.sidebar.write("LSTM：LSTM是一种在深度学习领域中使用的人工循环神经网络（RNN）架构。与标准前馈神经网络不同，LSTM具有反馈连接。它不仅可以处理单个数据点（例如图像），而且可以处理整个数据序列（例如语音或视频）。如果缺失值前的训练数据少于一定个数，则不能对缺失值进行预测。")

                elif methord2 == 'All' :
                    st.sidebar.write("所有预测算法：输出以上所有预测算法的预测结果。")
                #min = timeiter_fill_draw(df1)
                agree = st.sidebar.checkbox("下载", help="选择是否下载结果文件")
                agree2 = st.sidebar.button("开始", help="运行程序")



                if agree and agree2 :
                    # for col in list(df1) :
                    #     if col != value_col :
                    #         df1 = df1.drop([col], axis=1)
                    pdtime, mising, test = get_data_pd(df1)  # pdtime是标记过每个间隔的数据

                    # st.sidebar.write("**输出信息:**")
                    # st.sidebar.write("时间间隔: ", min, "  Min ")
                    from downstream import writecsv

                    if 1:
                        pdtime, mising, test = get_data_pd(df1)  # pdtime是标记过每个间隔的数据
                        st.sidebar.write("**输出信息：**")
                        st.sidebar.write("缺失值个数：", len(mising))  # 处理后数据有效值
                        st.sidebar.write("时间间隔: ", min, "  Min ")
                        with st.spinner('Wait for it...') :
                            bar = st.progress(0)
                            placeholder = st.empty()
                            placeholder.text("Load the histogram...")

                            time.sleep(0.1)
                            bar.progress(30)
                            placeholder.text("Load the time series figure...")

                            df2, fig = time_fill_csv(date_info, methord1, min, pdtime, mising, test,1)
                            time.sleep(0.1)
                            bar.progress(50)
                            with st.expander("查看完整图像 (注意:这可能会花费一些时间...)") :
                                if fig :
                                    st.plotly_chart(fig, use_container_width=True)
                            placeholder.text("Load the time series form...")
                            st.subheader('**Step 4**: 输出插值后的文件 :wave:')
                            st.write(df2.drop(['rownum'], axis=1).head(200))
                            bar.progress(70)
                            bar.progress(100)
                            placeholder.text('Finish...')
                        st.success("Done !")

                        with st.spinner('Wait for it...') :
                            #st.write(df2[methord1])
                            df_orig2 =df2[methord1].copy(deep=True)
                           
                            df_orig2.rename('value',inplace=True)
                           
                            
                            df_orig2=pd.DataFrame(df_orig2)
                  
                            
                           # st.write(df_orig2.values)
                           
                            pdtime, mising, test = get_data_pd2(df_orig2)
                           
                            bar = st.progress(0)
                            placeholder = st.empty()
                            placeholder.text("Load the histogram...")

                            time.sleep(0.1)
                            bar.progress(30)
                            placeholder.text("Load the time series figure...")
                           

                            df3, fig = time_pre_csv(date_info, methord2, min, pdtime, mising, test)
                            df3['missing']=df2['missing']
                            df3[df4['info'].iloc[0]]=""
                            # df3['info1'].iloc[0]=df4['info'].iloc[0]
                            # st.write(df4['info'].iloc[0])
                            # df3.rename({'info1':[df4['info'].iloc[0]]},inplace=True)

                           

                            time.sleep(0.1)
                            bar.progress(50)
                            with st.expander("查看完整图像 (注意:这可能会花费一些时间...)") :
                                if fig :
                                    st.plotly_chart(fig, use_container_width=True)
                            placeholder.text("Load the time series form...")
                            st.subheader('**Step 5**: 输出预测后的文件 :wave:')
                            
                            
                            bar.progress(70)
                            placeholder.text("Load the download url...")
                           
                            
                           

                            time1 = datetime.datetime.now()
                            now_time = datetime.datetime.strftime(time1, '%Y%m%d%H%M%S')
                            name = now_time + methord1+methord2 + ".csv"
                            kk = writecsv(df3, name)
                            st.dataframe(df3.head(200))
                            with st.expander("See  the entire output chart ") :
                                st.dataframe(df3)
                            st.subheader('**Step 6**: 文件下载链接 :wave:')
                            st.write('http://120.26.89.97:8501/downloads/' + name)
                            st.sidebar.write('http://120.26.89.97:8501/downloads/' + name)
                            st.sidebar.caption("生成文件的下载链接")
                            bar.progress(100)
                            placeholder.text('Finish...')
                        st.success("Done !")
                    else:
                        st.warning("ARIMA和LSTM模型的训练数据含有空值，请先选择插值！")

                elif agree2 :
                    pdtime, mising, test = get_data_pd(df1)  # pdtime是标记过每个间隔的数据

                    # st.sidebar.write("**输出信息:**")
                    # st.sidebar.write("时间间隔: ", min, "  Min ")


                    if 1:
                        pdtime, mising, test = get_data_pd(df1)  # pdtime是标记过每个间隔的数据
                        st.sidebar.write("**输出信息：**")
                        st.sidebar.write("缺失值个数：", len(mising))  # 处理后数据有效值
                        st.sidebar.write("时间间隔: ", min, "  Min ")
                        with st.spinner('Wait for it...') :
                            bar = st.progress(0)
                            placeholder = st.empty()
                            placeholder.text("Load the histogram...")

                            time.sleep(0.1)
                            bar.progress(30)
                            placeholder.text("Load the time series figure...")

                            df2, fig = time_fill_csv(date_info, methord1, min, pdtime, mising, test,1)
                            time.sleep(0.1)
                            bar.progress(50)
                            with st.expander("查看完整图像 (注意:这可能会花费一些时间...)") :
                                if fig :
                                    st.plotly_chart(fig, use_container_width=True)
                            placeholder.text("Load the time series form...")
                            st.subheader('**Step 4**: 输出插值后的文件 :wave:')
                            st.write(df2.drop(['rownum'], axis=1).head(200))
                            bar.progress(70)
                            bar.progress(100)
                            placeholder.text('Finish...')
                        st.success("Done !")

                        with st.spinner('Wait for it...') :
                            #st.write(df2[methord1])
                            df_orig2 =df2[methord1].copy(deep=True)
                           
                            df_orig2.rename('value',inplace=True)
                           
                            
                            df_orig2=pd.DataFrame(df_orig2)
                  
                            
                           # st.write(df_orig2.values)
                           
                            pdtime, mising, test = get_data_pd2(df_orig2)
                           
                            bar = st.progress(0)
                            placeholder = st.empty()
                            placeholder.text("Load the histogram...")

                            time.sleep(0.1)
                            bar.progress(30)
                            placeholder.text("Load the time series figure...")

                            df3, fig = time_pre_csv(date_info, methord2, min, pdtime, mising, test)
                            df3['missing']=df2['missing']
                            df3[df4['info'].iloc[0]]=""
                            time.sleep(0.1)
                            bar.progress(50)
                            with st.expander("查看完整图像 (注意:这可能会花费一些时间...)") :
                                if fig :
                                    st.plotly_chart(fig, use_container_width=True)
                            placeholder.text("Load the time series form...")
                            st.subheader('**Step 5**: 输出预测后的文件 :wave:')
                            st.dataframe(df3.head(200))
                            bar.progress(70)
                            
                            bar.progress(100)
                            placeholder.text('Finish...')
                        st.success("Done !")
                    else:
                        st.warning("ARIMA和LSTM模型的训练数据含有空值，请先选择插值！")
            if 1:
                if funn == '长期预测' :

                    st.subheader('**Step 3**: 您选择哪种插长期预测方法？ :wave:')

                    # st.sidebar.image("浙大.jpg")

                    methord4 = st.sidebar.selectbox(
                        '长期预测算法选择如下：',
                        ('LSTM_Predict', 'test1'
                         ))
                    st.write('您选择了：', methord4)
                    # if methord4 == 'Actual' :
                    #     st.sidebar.write("真实值：输出清洗过异常数据和重复数据，同时按照最大频次时间间隔切片处理的真实数据结果。")
                    if methord4 == 'LSTM_Predict' :
                        st.sidebar.write(
                            "LSTM：LSTM是一种在深度学习领域中使用的人工循环神经网络（RNN）架构。与标准前馈神经网络不同，LSTM具有反馈连接。它不仅可以处理单个数据点（例如图像），而且可以处理整个数据序列（例如语音或视频）。如果缺失值前的训练数据少于一定个数，则不能对缺失值进行预测。")
                    # elif methord3 == 'Backward_Fill' :
                    #     st.sidebar.write("后向插值：当前缺失值按照后一个已知值进行填充。如果缺失值后面没有真实值，则插值后仍未空值。")
                    # elif methord3 == 'Linear_Interpolation' :
                    #     st.sidebar.write("线性插值：通过连接相邻两个已知值的直线，以对在这两个已知值之间的缺失值进行填充。")
                    # elif methord3 == 'Cubic_Interpolation' :
                    #     st.sidebar.write(
                    #         "三次Hermite插值：通过在相邻两个已知值之间采用不超过三次的多项式函数将其连接，并且要求该函数二阶以下均可导的要求下，构建三次Hermite多项式，对当前缺失值进行插值。")
                    # elif methord3 == 'knn_mean' :
                    #     st.sidebar.write("最近邻插值：当前缺失值按照前4个时刻和后4个时刻已知值的平均值进行填充。如果前4个时刻和后4个时刻都为空值，则插值后仍未空值。")
                    # elif methord3 == 'test-50' :
                    #     st.sidebar.write("交叉验证：基于没有缺失的连续时间序列数据，对算法进行验证，输出每种插值算法的均方误差MSE。")
                    # elif methord3 == 'All' :
                    #     st.sidebar.write("所有插值算法：输出以上所有插值算法的插值结果。")
                    # min = timeiter_fill_draw(df1)
                    if methord4 != 'Actual' :
                        st.subheader('**Step 4**: 您选择如何划分训练集？ :wave:')

                        number = st.sidebar.number_input("请输入训练集所占比例：", min_value=0.5, max_value=0.9, value=0.8, step=0.1)
                        split = int(number * len(df1))
                        st.write("选择的数据集大小：", len(df1))
                        st.write("训练集大小：", split)
                        st.write("预测集大小：", len(df1) - split)

                    agree = st.sidebar.checkbox("下载", help="选择是否下载结果文件")
                    agree2 = st.sidebar.button("开始", help="运行程序")

                    if agree and agree2 :

                        if methord4 == 'LSTM_Predict' :
                            global df_p3
                            with st.spinner('Wait for it...') :
                                bar = st.progress(0)
                                placeholder = st.empty()
                                placeholder.text("Training model...")
                                if methord4 == 'LSTM_Predict' :

                                    #from model_lstm import read_items
                                    from testpost import runlstm
                                    #print(df1)

                                    result = runlstm(df1, split)
                                    result=str(result,'utf-8')
                                    result=eval(result)
                                    print(result)

                                    #result[0]["RMSE"] =float(result[0]["RMSE"])
                                    st.text("长短期记忆人工神经网络(LSTM) 均方根误差(RMSE):%.8f" % float(result["RMSE"]))

                                    df_predict = np.vstack(
                                        [result["predict"], result["real"]])  # shape = (10,6,7,8) 与上述效果相同
                                    df_predict = pd.DataFrame(df_predict.T)
                                    df_p2 = pd.DataFrame(result["predict"])
                                    df_p2.index = df1.index[-len(df_p2.index) :]
                                    df_p2.columns = ['LSTM_predict']

                                    df_p3 = pd.merge(df1, df_p2, left_index=True, right_index=True, how='left')

                                    df_p3.loc[df_p3.index.values[0], 'info'] = [""]

                                    df_p3.rename(columns={
                                        'info' : "输入特征数据是：" + str(value_col) + '\n' + "输出预测数据是：" + str(
                                            value_col2) + '\n' + "选择的数据集大小：" + str(len(df1)) + '\n' + "训练集大小：" + str(
                                            split) + '\n' + "测试集大小：" + str(
                                            len(df1) - split) + '\n' + "LSTM预测误差为：" + str(
                                            result['RMSE'])}, inplace=True)

                                    # st.write(df_p3)
                                    bar.progress(50)
                                    placeholder.text("Load the time series figure...")
                                    fig2 = go.Figure()
                                    fig2.add_trace(go.Scatter(
                                        name="Actual",
                                        mode="markers+lines", x=df_p2.head(1000).index, y=df_predict[1].head(1000),
                                        xperiodalignment="start"
                                    ))
                                    fig2.add_trace(go.Scatter(
                                        name="LSTM",
                                        mode="markers+lines", x=df_p2.head(1000).index, y=df_predict[0].head(1000),
                                        marker_symbol="star"
                                    ))

                                    fig2.update_layout(width=1000,
                                                       height=600,
                                                       xaxis_showgrid=False,
                                                       yaxis_showgrid=False,

                                                       showlegend=False)
                                    fig2.update_xaxes(
                                        showgrid=True,
                                        rangeslider_visible=True,
                                        rangeselector=dict(
                                            buttons=list([
                                                dict(count=1, label="1小时", step="hour", stepmode="backward"),
                                                dict(count=1, label="1天", step="day", stepmode="backward"),
                                                dict(count=1, label="1个月", step="month", stepmode="todate"),
                                                dict(count=1, label="1年", step="year", stepmode="backward"),
                                                dict(step="all")
                                            ])
                                        )
                                    )
                                    st.plotly_chart(fig2, use_container_width=True)

                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        name="Actual",
                                        mode="markers+lines", x=df_p2.index, y=df_predict[1].head(1000),
                                        xperiodalignment="start"
                                    ))
                                    fig.add_trace(go.Scatter(
                                        name="LSTM",
                                        mode="markers+lines", x=df_p2.index, y=df_predict[0].head(1000),
                                        marker_symbol="star"
                                    ))

                                    fig.update_layout(width=1000,
                                                      height=600,
                                                      xaxis_showgrid=False,
                                                      yaxis_showgrid=False,

                                                      showlegend=False)
                                    fig.update_xaxes(
                                        showgrid=True,
                                        rangeslider_visible=True,
                                        rangeselector=dict(
                                            buttons=list([
                                                dict(count=1, label="1小时", step="hour", stepmode="backward"),
                                                dict(count=1, label="1天", step="day", stepmode="backward"),
                                                dict(count=1, label="1个月", step="month", stepmode="todate"),
                                                dict(count=1, label="1年", step="year", stepmode="backward"),
                                                dict(step="all")
                                            ])
                                        )
                                    )
                                    with st.expander("查看完整图像 (注意:这可能会花费一些时间...)") :
                                        if fig :
                                            st.plotly_chart(fig, use_container_width=True)
                                    bar.progress(70)
                                    placeholder.text("Load the data...")

                                time1 = datetime.datetime.now()
                                now_time = datetime.datetime.strftime(time1, '%Y%m%d%H%M%S')
                                name = now_time + methord4 + value_col2 + ".csv"
                                kk = writecsv(df_p3, name)
                                st.dataframe(df_p3.head(200))
                                with st.expander("See  the entire output chart ") :
                                    st.dataframe(df_p3)
                                st.subheader('**Step 5**: 文件下载链接 :wave:')
                                st.write('http://120.26.89.97:8501/downloads/' + name)
                                st.sidebar.write('http://120.26.89.97:8501/downloads/' + name)
                                st.sidebar.caption("生成文件的下载链接")
                                bar.progress(100)
                                placeholder.text('Finish...')
                            st.success("Done !")

                    elif agree2 :

                        with st.spinner('Wait for it...') :
                            bar = st.progress(0)
                            placeholder = st.empty()
                            placeholder.text("Training model...")
                            if methord4 == 'LSTM_Predict' :

                                from model_lstm import read_root

                                result = dict(read_root(df1, split))
                                st.text("长短期记忆人工神经网络(LSTM) 均方根误差(RMSE):%.8f" % result["RMSE"])

                                df_predict = np.vstack(
                                    [result["predict"], result["real"]])  # shape = (10,6,7,8) 与上述效果相同
                                df_predict = pd.DataFrame(df_predict.T)
                                df_p2 = pd.DataFrame(result["predict"])
                                df_p2.index = df1.index[-len(df_p2.index) :]
                                df_p2.columns = ['LSTM_predict']

                                df_p3 = pd.merge(df1, df_p2, left_index=True, right_index=True, how='left')

                                df_p3.loc[df_p3.index.values[0], 'info'] = [""]

                                df_p3.rename(columns={
                                    'info' : "输入特征数据是：" + str(value_col) + '\n' + "输出预测数据是：" + str(
                                        value_col2) + '\n' + "选择的数据集大小：" + str(len(df1)) + '\n' + "训练集大小：" + str(
                                        split) + '\n' + "测试集大小：" + str(len(df1) - split) + '\n' + "LSTM预测误差为：" + str(
                                        result['RMSE'])}, inplace=True)

                                # st.write(df_p3)
                                bar.progress(50)
                                placeholder.text("Load the time series figure...")
                                fig2 = go.Figure()
                                fig2.add_trace(go.Scatter(
                                    name="Actual",
                                    mode="markers+lines", x=df_p2.head(1000).index, y=df_predict[1].head(1000),
                                    xperiodalignment="start"
                                ))
                                fig2.add_trace(go.Scatter(
                                    name="LSTM",
                                    mode="markers+lines", x=df_p2.head(1000).index, y=df_predict[0].head(1000),
                                    marker_symbol="star"
                                ))

                                fig2.update_layout(width=1000,
                                                   height=600,
                                                   xaxis_showgrid=False,
                                                   yaxis_showgrid=False,

                                                   showlegend=False)
                                fig2.update_xaxes(
                                    showgrid=True,
                                    rangeslider_visible=True,
                                    rangeselector=dict(
                                        buttons=list([
                                            dict(count=1, label="1小时", step="hour", stepmode="backward"),
                                            dict(count=1, label="1天", step="day", stepmode="backward"),
                                            dict(count=1, label="1个月", step="month", stepmode="todate"),
                                            dict(count=1, label="1年", step="year", stepmode="backward"),
                                            dict(step="all")
                                        ])
                                    )
                                )
                                st.plotly_chart(fig2, use_container_width=True)

                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    name="Actual",
                                    mode="markers+lines", x=df_p2.index, y=df_predict[1].head(1000),
                                    xperiodalignment="start"
                                ))
                                fig.add_trace(go.Scatter(
                                    name="LSTM",
                                    mode="markers+lines", x=df_p2.index, y=df_predict[0].head(1000),
                                    marker_symbol="star"
                                ))

                                fig.update_layout(width=1000,
                                                  height=600,
                                                  xaxis_showgrid=False,
                                                  yaxis_showgrid=False,

                                                  showlegend=False)
                                fig.update_xaxes(
                                    showgrid=True,
                                    rangeslider_visible=True,
                                    rangeselector=dict(
                                        buttons=list([
                                            dict(count=1, label="1小时", step="hour", stepmode="backward"),
                                            dict(count=1, label="1天", step="day", stepmode="backward"),
                                            dict(count=1, label="1个月", step="month", stepmode="todate"),
                                            dict(count=1, label="1年", step="year", stepmode="backward"),
                                            dict(step="all")
                                        ])
                                    )
                                )
                                with st.expander("查看完整图像 (注意:这可能会花费一些时间...)") :
                                    if fig :
                                        st.plotly_chart(fig, use_container_width=True)
                                bar.progress(70)
                                placeholder.text("Load the data...")

                            st.dataframe(df_p3.head(200))
                            with st.expander("See  the entire output chart ") :
                                st.dataframe(df_p3)
                            bar.progress(100)
                            placeholder.text('Finish...')
                        st.success("Done !")
            else:
                st.warning("请先检查或处理文件数据！")


            st.sidebar.caption("Made by ruru@zju")

    else:
        st.warning("**请先上传文件！**")
        st.stop()

    # except:
    #     st.warning("**Please try refreshing the page !**")
    #     st.stop()


