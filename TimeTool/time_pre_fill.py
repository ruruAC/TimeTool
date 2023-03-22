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
import random  # test 1/5的测试集
import time
from timeinter import timeiter_fill, timeiter_fill_draw

import streamlit as st

import numpy as np


def create_dataset(dataset, look_back) :
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1) :
        a = dataset[i :(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

def autoNorm(data) :  # 传入一个矩阵
    mins = data.min(0)  # 返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0)  # 返回data矩阵中每一列中最大的元素，返回一个列表
    ranges = maxs - mins  # 最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data))  # 生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0]  # 返回 data矩阵的行数
    normData = data - np.tile(mins, (row, 1))  # data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(ranges, (row, 1))  # data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData


max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))

def time_pre_csv(date_info, methord, min, pdtime, mising, test) :
    qs = pdtime['value'].isnull().sum(axis=0)
    st.sidebar.write("预测值个数：", qs)
    # 根据序列的内在属性以下是几种比较有效的填充方法：

    # 向后填充法
    # 线性插值法
    # 三次H插值法
    # 最近邻均值法

    df = pdtime.copy(deep=True)
    num_test = 50
    if methord=="Actual":
            df.loc[df.index.values[0], 'predict_info'] = [""]
            df.rename(columns={'predict_info':"“Value”异常个数：%d:" % date_info[1] + '\n' + "“Date”异常个数：%d；" % date_info[3] + '\n' + "“Date”重复个数：%d；" %
        date_info[2] + '\n' + "缺失值总数为:%d；" % qs \
        + '\n' + "有效值总数为:%d；" % date_info[4] + '\n'},inplace=True)



    ## 1. Actual -------------------------------
    if methord == 'Actual' or methord == 'All' :

        # df_orig.plot(label='Actual', linewidth=3.0)
        # plt.legend(["Available Data"])
        # plt.show()

        if methord != 'All' :
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
    ## 2.Moving_Average' --------------------------
    def test_ff() :
        ss=0
        l=0
        val['Moving Average']=val['value'] #需要选定一项插值方法
        for i in range(len(val)):
            if np.isnan(val['Moving Average'][i]):
                if l!=0:
                    val['Moving Average'][i]=ss/l
            else:
                ss=ss+val['Moving Average'][i]
                l=l+1
        rmse = np.sqrt(np.mean(np.power((np.array(test['value']) - np.array(val['Moving Average'])), 2)))

       # error = np.round(mean_squared_error(test['value'], val['Moving Average']) * num_test / (int(num_test / 5)), 8)

        return rmse
    def testMA():
        ff = []

        for t in range(0, 5) :
            global val
            val = test.copy(deep=True)
            alist = random.sample(range(1, num_test), int(num_test / 5))  # random.sample()生成不相同的随机数
            val.value[alist] = np.NaN

            ff.append(test_ff())
        st.sidebar.write(" 移动平均（Moving Average） 均方根误差（RMSE):" ,mean(ff))
        return mean(ff)
    global rmselist
    rmselist=[]


    ## 2. Moving Average --------------------------
    if methord == 'Moving_Average' or methord == 'All' :
        wc=testMA()
        ss=0
        l=0
        # qs = df['value'].isnull().sum(axis=0)
        # st.sidebar.write("预测值个数：", qs)
        df['Moving Average']=df['value'] #需要选定一项插值方法
        for i in range(len(df)):
            if np.isnan(df['Moving Average'][i]):
                if l!=0:
                    df['Moving Average'][i]=ss/l
            else:
                ss=ss+df['Moving Average'][i]
                l=l+1
        #st.sidebar.write("预测值个数：", qs)


        rmselist.append(wc)


        if methord != 'All' :
            df.loc[df.index.values[0], 'predict_info'] = [""]
            df.rename(columns={'predict_info':"“Value”异常个数：%d:" % date_info[1] + '\n' + "“Date”异常个数：%d；" % date_info[3] + '\n' + "“Date”重复个数：%d；" %
            date_info[2] + '\n' + "缺失值总数为:%d；" % qs \
            + '\n' + "有效值总数为:%d；" % date_info[4] + '\n' \
            + "移动平均（Moving Average） 均方根误差（RMSE): %.8f。" % wc},inplace=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.head(1000).index, y=pdtime["value"].head(1000),
                xperiodalignment="start"
            ))

            fig2.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.head(1000).index, y=df['Moving Average'].head(1000),
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
                mode="markers+lines", x=pdtime.index, y=pdtime["value"],
                xperiodalignment="start"
            ))

            fig.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.index, y=df['Moving Average'],
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
            st.line_chart(df[['value','Moving Average']])

    ## 3. LinearRegression -------------------------
    if methord == 'Linear_Regression' or methord == 'All' :
        type='value'
        new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', type])

        for i in range(0, len(df)) :
            new_data['Date'][i] = df.index[i]
            new_data[type][i] = df[type][i]

        # create features
        from fastai.tabular.transform import add_datepart
        add_datepart(new_data, 'Date')
        new_data['timestamp'] = 0

        d = new_data['Day'][0]
        new_data['timestamp'][0]=0
        xh = 1
        for i in range(1, len(new_data)) :

            if (new_data['Day'][i] == d ) :  
                new_data['timestamp'][i] = xh
                xh=xh+1
            else :
                d=new_data['Day'][i]
                xh=1
                new_data['timestamp'][i]=0

        #new_data.drop('Elapsed', axis=1, inplace=True)  # elapsed will be the time stamp
        # new_data.drop('Year', axis=1, inplace=True)  # elapsed will be the time stamp

        #归一化
        new_data['Day']=new_data[['Day']].apply(max_min_scaler)
        new_data['Year'] = new_data[['Year']].apply(max_min_scaler)
        new_data['Week']=new_data[['Week']].apply(max_min_scaler)
        new_data['Dayofweek']=new_data[['Dayofweek']].apply(max_min_scaler)
        new_data['Dayofyear']=new_data[['Dayofyear']].apply(max_min_scaler)
        new_data['timestamp'] = new_data[['timestamp']].apply(max_min_scaler)
        new_data.drop(new_data.columns[[13]], axis=1, inplace=True)  #'Elapsed'


        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        #st.write(new_data)

        train_data=new_data.dropna(subset=[type]) #删去value 为空的行
        
        train_data.fillna(1, inplace=True) #如果除了value列，其他列有空值的需要填充，不然也会被当成空值删掉
     
        model.fit(train_data.drop(type,axis=1),train_data[type])



        preds = model.predict(train_data.drop(type,axis=1).fillna(1))
        rmse = np.sqrt(np.mean(np.power((np.array(train_data['value']) - np.array(preds)), 2)))

        #st.text(" 线性回归（Linear Regression） 均方根误差（RMSE）:%.8f" % rmse)
        new_data.drop(type, axis=1)
        
        pred=model.predict(new_data.drop(type, axis=1).fillna(1))
       # st.write(new_data.drop(type, axis=1).fillna(1))
        
        qs=0

        df['Linear Regression']=df['value'] #需要选定一项插值方法
        for i in range(len(df)):
            if np.isnan(df['Linear Regression'][i]) :
                qs=qs+1
                df['Linear Regression'][i]=pred[i]
        #st.sidebar.write("预测值个数：",qs)
        st.sidebar.write("线性回归（Linear Regression）均方根误差(RMSE):", rmse)
       


        rmselist.append(rmse)

        if methord != 'All' :
            df.loc[df.index.values[0], 'predict_info'] = [""]
            df.rename(columns={'predict_info':"“Value”异常个数：%d:" % date_info[1] + '\n' + "“Date”异常个数：%d；" % date_info[3] + '\n' + "“Date”重复个数：%d；" %
            date_info[2] + '\n' + "缺失值总数为:%d；" % qs \
            + '\n' + "有效值总数为:%d；" % date_info[4] + '\n' \
            + "线性回归（Linear Regression） 均方根误差（RMSE): %.8f。" % rmse},inplace=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.head(1000).index, y=pdtime["value"].head(1000),
                xperiodalignment="start"
            ))
            fig2.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.head(1000).index, y=df['Linear Regression'].head(1000),
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
                mode="markers+lines", x=pdtime.index, y=df['Linear Regression'],
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
            st.line_chart(df[['value','Linear Regression']])


    ## 4. K_Nearest_Neighbours ------------------ 从一个NA的相邻两个真实点(相邻就是最近的两个点，如果是两边正好平均，否则就是一侧的两个最近点） 代入y=ax+b
    if methord == 'K_Nearest_Neighbours' or methord == 'All' :
        type='value'
        new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', type])

        for i in range(0, len(df)) :
            new_data['Date'][i] = df.index[i]
            new_data[type][i] = df[type][i]

        # create features
        from fastai.tabular.transform import add_datepart
        add_datepart(new_data, 'Date')
        new_data['timestamp'] = 0

        d = new_data['Day'][0]
        new_data['timestamp'][0] = 0
        xh = 1
        for i in range(1, len(new_data)) :

            if (new_data['Day'][i] == d) :
                new_data['timestamp'][i] = xh
                xh = xh + 1
            else :
                d = new_data['Day'][i]
                xh = 1
                new_data['timestamp'][i] = 0

        # new_data.drop('Elapsed', axis=1, inplace=True)  # elapsed will be the time stamp
        # new_data.drop('Year', axis=1, inplace=True)  # elapsed will be the time stamp

        # 归一化
        new_data['Day'] = new_data[['Day']].apply(max_min_scaler)
        new_data['Year'] = new_data[['Year']].apply(max_min_scaler)
        new_data['Week'] = new_data[['Week']].apply(max_min_scaler)
        new_data['Dayofweek'] = new_data[['Dayofweek']].apply(max_min_scaler)
        new_data['Dayofyear'] = new_data[['Dayofyear']].apply(max_min_scaler)
        new_data['timestamp'] = new_data[['timestamp']].apply(max_min_scaler)
        new_data.drop(new_data.columns[[13]], axis=1, inplace=True)  # 'Elapsed'

        # importing libraries
        from sklearn import neighbors
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import MinMaxScaler
        # using gridsearch to find the best parameter
        params = {'n_neighbors' : [2, 3, 4, 5, 6, 7, 8, 9]}  #邻居个数,每个样本都可以用最接近的k个邻居表示，或者说共享同一个标签
        knn = neighbors.KNeighborsRegressor()

        model = GridSearchCV(knn, params, cv=5)
        # fit the model and make predictions

        train_data=new_data.dropna(subset=[type]) #删去value 为空的行
        train_data.fillna(1, inplace=True) #如果除了value列，其他列有空值的需要填充，不然也会被当成空值删掉

        model.fit(train_data.drop(type,axis=1),train_data[type])

        preds = model.predict(train_data.drop(type,axis=1))
        rmse = np.sqrt(np.mean(np.power((np.array(train_data['value']) - np.array(preds)), 2)))




        pred=model.predict(new_data.drop(type, axis=1).fillna(1))
       # st.write(new_data.drop(type, axis=1).fillna(1))
        

        qs=0
        rmselist.append(rmse)

        df['K_Nearest_Neighbours']=df['value'] #需要选定一项插值方法
        for i in range(len(df)):
            if np.isnan(df['K_Nearest_Neighbours'][i]) :
                qs=qs+1
                df['K_Nearest_Neighbours'][i]=pred[i]
        #st.sidebar.write("预测值个数：",qs)
        st.sidebar.write("K Nearest Neighbours均方根误差(RMSE):", rmse)
        #print(model.best_params_)

        # rmse




        if methord != 'All' :
            df.loc[df.index.values[0], 'predict_info'] = [""]
            df.rename(columns={'predict_info':
            "“Value”异常个数：%d:" % date_info[1] + '\n' + "“Date”异常个数：%d；" % date_info[3] + '\n' + "“Date”重复个数：%d；" %
            date_info[2] + '\n' + "缺失值总数为:%d；" % qs \
            + '\n' + "有效值总数为:%d；" % date_info[4] + '\n' \
            + "K Nearest Neighbours的均方根误差(RMSE): %.8f。" % rmse},inplace=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.head(1000).index, y=pdtime["value"].head(1000),
                xperiodalignment="start"
            ))

            fig2.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.head(1000).index, y=df['K_Nearest_Neighbours'].head(1000),
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
                mode="markers+lines", x=pdtime.index, y=df['K_Nearest_Neighbours'],
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
            st.line_chart(df[['value','K_Nearest_Neighbours']])



    def testAR():
        from pmdarima import auto_arima
        type='value'
        train = test[:30]
        valid = test[30:]

        training = train[type]
        validation = valid[type]


        model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                           trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(training)

        forecast = model.predict(20)
        forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])


        rmse = np.sqrt(np.mean(np.power((np.array(valid[type]) - np.array(forecast['Prediction'])), 2)))
        st.sidebar.write("差分整合移动平均自回归模型（ARIMA） 均方根误差(RMSE):",rmse)
        return rmse
    ## 5. ARIMA -------------------- 三次Hermite插值，采用H多项式，在斜率选择上 略微不同于三次样本插值



    if methord == 'ARIMA' or methord == 'All' :
        wc=testAR()
        from pmdarima import auto_arima
        type='value'

        # 设置滚动预测的参数

        df['ARIMA'] = df[type]  # 需要选定一项插值方法
        ts = df['ARIMA']
        test_size = sum(ts.isnull() == True)
#判断第一行为空值的个数，如果找非空值个数，则将True改为False
  # 需要预测的个数
        rolling_size = 120  # 滚动窗口大小
        ps = 1  # 每次预测的个数
        horizon = 1  # 用来消除切片的影响
        train_data = df.dropna()


        qs=0
        pre=[]

        test_size2=len(df)-test_size


        # 滚动预测

        with st.spinner('Wait for ARIMA training...') :
            
            bar = st.progress(0)
            jindu=0
            

            for i in range(30,len(df)) :

                if df['ARIMA'][i]!=df['ARIMA'][i]: #不等于自身，是空值

                    qs = qs + 1
                    if i<=120:
                        w=0
                    else:
                        w=i-120
                    train=df[w:i]
                    print("______",i)
            
                    train=train['ARIMA']
              

                    model = auto_arima(train, start_p=0, start_q=0, max_p=6, max_q=6, max_d=2,
                                       seasonal=False, test='adf',
                                       error_action='ignore',
                                       information_criterion='aic',
                                       njob=-1, suppress_warnings=True)
                    #st.write(model)


                    model.fit(train)

                    forecast = model.predict(n_periods=ps)
                    df['ARIMA'][i] = forecast[-1]

                if i % (int((len(df) / 10))) == 0 and jindu < 100 :
                    jindu = jindu + 10
                    bar.progress(jindu)
            bar.progress(100)
            st.text("Training Done !")



        '''
            网址:http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html?highlight=auto_arima
            auto_arima部分参数解析:
                1.start_p:p的起始值，自回归(“AR”)模型的阶数(或滞后时间的数量),必须是正整数
                2.start_q:q的初始值，移动平均(MA)模型的阶数。必须是正整数。
                3.max_p:p的最大值，必须是大于或等于start_p的正整数。
                4.max_q:q的最大值，必须是一个大于start_q的正整数
                5.seasonal:是否适合季节性ARIMA。默认是正确的。注意，如果season为真，而m == 1，则season将设置为False。
                6.stationary :时间序列是否平稳，d是否为零。
                6.information_criterion：信息准则用于选择最佳的ARIMA模型。(‘aic’，‘bic’，‘hqic’，‘oob’)之一
                7.alpha：检验水平的检验显著性，默认0.05
                8.test:如果stationary为假且d为None，用来检测平稳性的单位根检验的类型。默认为‘kpss’;可设置为adf
                9.n_jobs ：网格搜索中并行拟合的模型数(逐步=False)。默认值是1，但是-1可以用来表示“尽可能多”。
                10.suppress_warnings：statsmodel中可能会抛出许多警告。如果suppress_warnings为真，那么来自ARIMA的所有警告都将被压制
                11.error_action:如果由于某种原因无法匹配ARIMA，则可以控制错误处理行为。(warn,raise,ignore,trace)
                12.max_d:d的最大值，即非季节差异的最大数量。必须是大于或等于d的正整数。
                13.trace:是否打印适合的状态。如果值为False，则不会打印任何调试信息。值为真会打印一些
            '''


        #st.sidebar.write("预测值个数：", qs)


        rmselist.append(wc)



        if methord != 'All' :
            df.loc[df.index.values[0], 'predict_info'] = [""]
            df.rename(columns={'predict_info':"“Value”异常个数：%d:" % date_info[1] + '\n' + "“Date”异常个数：%d；" % date_info[3] + '\n' + "“Date”重复个数：%d；" %
            date_info[2] + '\n' + "缺失值总数为:%d；" % qs \
            + '\n' + "有效值总数为:%d；" % date_info[4] + '\n' \
            + "差分整合移动平均自回归模型（ARIMA）的均方根误差(RMSE): %.8f。" % wc},inplace=True)
            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.head(1000).index, y=pdtime['value'].head(1000),
                xperiodalignment="start"
            ))
            fig2.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.head(1000).index, y=df['ARIMA'].head(1000),
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
                mode="markers+lines", x=pdtime.index, y=df['ARIMA'],
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
    def testLS():
        import numpy
        import matplotlib.pyplot as plt
        from pandas import read_csv
        import math
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM

        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error
        import os

        type = 'value'
        new_data = pd.DataFrame(index=range(0, len(test)), columns=['Date', type])

        for i in range(0, len(test)) :
            new_data['Date'][i] = test.index[i]
            new_data[type][i] = test[type][i]
        new_data.index = new_data.Date
        new_data.drop('Date', axis=1, inplace=True)

        # 上面代码的片段讲解
        scaler = MinMaxScaler(feature_range=(0, 1))

        new_data = scaler.fit_transform(new_data)
        look_back = 1  # 预测几个值

        trainX, trainY = create_dataset(new_data, look_back)

        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        trainX = trainX.astype('float32')
        trainY = trainY.astype('float32')

        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))  # 1个输入，1个输出，4个隐藏
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
        trainPredict = model.predict(trainX)
        trainPredict = scaler.inverse_transform(trainPredict)
        rmse = np.sqrt(np.mean(np.power((np.array(trainX) - np.array(trainPredict)), 2)))
        st.sidebar.write("长短期记忆人工神经网络(LSTM) 均方根误差(RMSE):", rmse)
        return rmse
        
    if methord == 'LSTM':

   # if methord == 'LSTM' or methord == 'All' :
   
        jindu=0



        wc=testLS()
        import numpy
        import matplotlib.pyplot as plt
        from pandas import read_csv
        import math
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense,LSTM

        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error
        import os

        type = 'value'
        new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', type])

        for i in range(0, len(df)) :
            new_data['Date'][i] = df.index[i]
            new_data[type][i] = df[type][i]
        new_data.index = new_data.Date
        new_data.drop('Date', axis=1, inplace=True)
        

        # 上面代码的片段讲解
        scaler = MinMaxScaler(feature_range=(0, 1))


        df['LSTM']=df[type]
        new_data = scaler.fit_transform(new_data)
        look_back = 1  #预测几个值

        
        qs=0
        with st.spinner('Wait for LSTM training...') :
            bar = st.progress(0)
            fs=int(len(new_data)/10)
            for i in range(30,len(new_data)) :

                if df['LSTM'][i]!=df['LSTM'][i] :
                    if i <= 120 :
                        w = 0
                    else :
                        w = i - 120
                    train_data2 = new_data[w :i]


                    qs=qs+1

                    trainX, trainY = create_dataset(train_data2, look_back)
                    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                    trainX = trainX.astype('float32')
                    trainY = trainY.astype('float32')

                    model = Sequential()
                    model.add(LSTM(4, input_shape=(1, look_back)))  # 1个输入，1个输出，4个隐藏
                    model.add(Dense(1))
                    model.compile(loss='mean_squared_error', optimizer='adam')
                    model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
                    trainPredict = model.predict(trainX[-1 :])
                    new_data[i]=trainPredict
                    trainPredict = scaler.inverse_transform(trainPredict)
                    df['LSTM'][i] = trainPredict
                    #train_data.append(df['LSTM'][i])

                if i%(fs)==0 and jindu<100:
                    jindu = jindu + 10
                    bar.progress(jindu)
            bar.progress(100)
            st.text("Training Done !")
        #st.sidebar.write("预测值个数：", qs)


        rmselist.append(wc)



        if methord != 'All' :
            df.loc[df.index.values[0], 'predict_info'] = [""]
            df.rename(columns={'predict_info':"“Value”异常个数：%d:" % date_info[1] + '\n' + "“Date”异常个数：%d；" % date_info[3] + '\n' + "“Date”重复个数：%d；" %
            date_info[2] + '\n' + "缺失值总数为:%d；" % qs \
            + '\n' + "有效值总数为:%d；" % date_info[4] + '\n' \
            + "长短期记忆人工神经网络(LSTM) 均方根误差(RMSE):%.8f。" % wc},inplace=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name="Actual",
                mode="markers+lines", x=pdtime.head(1000).index, y=pdtime["value"].head(1000),
                xperiodalignment="start"
            ))
            fig2.add_trace(go.Scatter(
                name=methord,
                mode="markers+lines", x=pdtime.head(1000).index, y=df['LSTM'].head(1000),
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
                mode="markers+lines", x=pdtime.index, y=df['LSTM'],
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

    # print("缺失值个数：", len(mising))
    if methord == 'All' :
       # print("插值方法为：'Forward Fill', 'Backward Fill', 'Linear Interpolation', 'Cubic Interpolation','knn_mean'")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            name="Actual",
            mode="markers+lines", x=df.head(1000).index, y=df["value"].head(1000),
            xperiodalignment="start"
        ))

        fig2.add_trace(go.Scatter(
            name='Moving Average',
            mode="markers+lines", x=df.head(1000).index, y=df['Moving Average'].head(1000),
            marker_symbol="star"
        ))
        fig2.add_trace(go.Scatter(
            name='Linear Regression',
            mode="markers+lines", x=df.head(1000).index, y=df['Linear Regression'].head(1000),
            xperiodalignment="start"
        ))
        fig2.add_trace(go.Scatter(
            name='K Nearest Neighbours',
            mode="markers+lines", x=df.head(1000).index, y=df['K_Nearest_Neighbours'].head(1000),
            xperiodalignment="start"
        ))
        fig2.add_trace(go.Scatter(
            name='ARIMA',
            mode="markers+lines", x=df.head(1000).index, y=df['ARIMA'].head(1000),
            xperiodalignment="start"
        ))
        
        # fig2.add_trace(go.Scatter(
        #     name='LSTM',
        #     mode="markers+lines", x=df.head(1000).index, y=df['LSTM'].head(1000),
        #     xperiodalignment="start"
        # ))

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
            name='Moving Average',
            mode="markers+lines", x=df.index, y=df['Moving Average'],
            marker_symbol="star"
        ))
        fig.add_trace(go.Scatter(
            name='Linear Regression',
            mode="markers+lines", x=df.index, y=df['Linear Regression'],
            xperiodalignment="start"
        ))
        fig.add_trace(go.Scatter(
            name='K Nearest Neighbours',
            mode="markers+lines", x=df.index, y=df['K_Nearest_Neighbours'],
            xperiodalignment="start"
        ))
        fig.add_trace(go.Scatter(
            name='ARIMA',
            mode="markers+lines", x=df.index, y=df['ARIMA'],
            xperiodalignment="start"
        ))
        # fig.add_trace(go.Scatter(
        #     name='LSTM',
        #     mode="markers+lines", x=df.index, y=df['LSTM'],
        #     xperiodalignment="start"
        # ))

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



    ## 3. Linear Regression -------------------------
    def test_bf() :
        type = 'value'
        new_data = pd.DataFrame(index=range(0, len(val)), columns=['Date', type])

        for i in range(0, len(val)) :
            new_data['Date'][i] = val.index[i]
            new_data[type][i] = val[type][i]

        # create features
        from fastai.tabular.transform import add_datepart
        add_datepart(new_data, 'Date')
        new_data['timestamp'] = 0

        d = new_data['Day'][0]
        new_data['timestamp'][0] = 0
        xh = 1
        for i in range(1, len(new_data)) :

            if (new_data['Day'][i] == d) :  # 如果是星期一或星期五
                new_data['timestamp'][i] = xh
                xh = xh + 1
            else :
                d = new_data['Day'][i]
                xh = 1
                new_data['timestamp'][i] = 0

        # new_data.drop('Elapsed', axis=1, inplace=True)  # elapsed will be the time stamp
        # new_data.drop('Year', axis=1, inplace=True)  # elapsed will be the time stamp

        # 归一化
        new_data['Day'] = new_data[['Day']].apply(max_min_scaler)
        new_data['Year'] = new_data[['Year']].apply(max_min_scaler)
        new_data['Week'] = new_data[['Week']].apply(max_min_scaler)
        new_data['Dayofweek'] = new_data[['Dayofweek']].apply(max_min_scaler)
        new_data['Dayofyear'] = new_data[['Dayofyear']].apply(max_min_scaler)
        new_data['timestamp'] = new_data[['timestamp']].apply(max_min_scaler)
        new_data.drop(new_data.columns[[1,3,13]], axis=1, inplace=True)  # 'Elapsed'
        #new_data.drop(new_data.columns['Year','Week'], axis=1, inplace=True)  # 'Elapsed'

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        train_data=new_data.dropna(axis=0, subset=(type, ))

        model.fit(train_data.drop(type, axis=1), train_data[type])

        preds = model.predict(new_data.drop(type, axis=1))
        rmse = np.sqrt(np.mean(np.power((np.array(test['value']) - np.array(preds)), 2)))

       # error = np.round(mean_squared_error(test['value'], train_data['value']) * num_test / (int(num_test / 5)), 8)

        return rmse

    ## 4. 'K_Nearest_Neighbours'  ------------------
    def test_li() :
        type = 'value'
        new_data = pd.DataFrame(index=range(0, len(val)), columns=['Date', type])

        for i in range(0, len(val)) :
            new_data['Date'][i] = df.index[i]
            new_data[type][i] = df[type][i]

        # create features
        from fastai.tabular.transform import add_datepart
        add_datepart(new_data, 'Date')
        new_data['timestamp'] = 0

        d = new_data['Day'][0]
        new_data['timestamp'][0] = 0
        xh = 1
        for i in range(1, len(new_data)) :

            if (new_data['Day'][i] == d) :
                new_data['timestamp'][i] = xh
                xh = xh + 1
            else :
                d = new_data['Day'][i]
                xh = 1
                new_data['timestamp'][i] = 0

        # new_data.drop('Elapsed', axis=1, inplace=True)  # elapsed will be the time stamp
        # new_data.drop('Year', axis=1, inplace=True)  # elapsed will be the time stamp

        # 归一化
        new_data['Day'] = new_data[['Day']].apply(max_min_scaler)
        new_data['Year'] = new_data[['Year']].apply(max_min_scaler)
        new_data['Week'] = new_data[['Week']].apply(max_min_scaler)
        new_data['Dayofweek'] = new_data[['Dayofweek']].apply(max_min_scaler)
        new_data['Dayofyear'] = new_data[['Dayofyear']].apply(max_min_scaler)
        new_data['timestamp'] = new_data[['timestamp']].apply(max_min_scaler)
        new_data.drop(new_data.columns[[1,3,13]], axis=1, inplace=True)

        # importing libraries
        from sklearn import neighbors
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import MinMaxScaler
        # using gridsearch to find the best parameter
        params = {'n_neighbors' : [2, 3, 4, 5, 6, 7, 8, 9]}  # 邻居个数,每个样本都可以用最接近的k个邻居表示，或者说共享同一个标签
        knn = neighbors.KNeighborsRegressor()

        model = GridSearchCV(knn, params, cv=5)
        # fit the model and make predictions

        train_data = new_data.dropna()

        model.fit(train_data.drop(type, axis=1), train_data[type])

        preds = model.predict(new_data.drop(type, axis=1))
        rmse = np.sqrt(np.mean(np.power((np.array(new_data[type]) - np.array(preds)), 2)))

        return rmse

    if  methord == 'All' :

        for t in range(0, 5) :
            global val
            val = test.copy(deep=True)
            alist = random.sample(range(1, num_test), int(num_test / 5))  # random.sample()生成不相同的随机数
            val.value[alist] = np.nan

        # 求每个列表平均值 返回最大那个，可以用遍历加索引，然后得到最优的方法
        # 然后用最优的方法去插值所有
        # 连续缺失值应该有一个标记 后者直接knn nan状态就是异常的
        # 也可以全部方法都用 然后再可以验证
        qs = df['value'].isnull().sum(axis=0)

        st.subheader("每种方法均方差比较：")

        st.text("移动平均（Moving Average） 均方根误差（RMSE): %.8f；" % rmselist[0])
        st.text("线性回归（Linear Regression） 均方根误差（RMSE): %.8f；" % rmselist[1])
        st.text("最近邻（K-Nearest Neighbours）均方根误差(RMSE): %.8f；" % rmselist[2])
        st.text("差分整合移动平均自回归模型（ARIMA） 均方根误差(RMSE):%.8f；" % rmselist[3])
        #st.text("长短期记忆人工神经网络(LSTM) 均方根误差(RMSE):%.8f" % rmselist[4])
        df.loc[df.index.values[0], 'predict_info'] = [""]
        df.rename(columns={'predict_info':"“Value”异常个数：%d；" % date_info[1] + '\n' + "“Date”异常个数：%d；" % date_info[3] + '\n' + "“Date”重复个数：%d；" %
            date_info[2] + '\n' + "缺失值总数为:%d；" % qs \
            + '\n' + "有效值总数为:%d；" % date_info[4] + '\n' + '每种方法均方差：' + '\n' + "移动平均（Moving Average） 均方根误差（RMSE): %.8f；" % rmselist[0] + '\n' + "线性回归（Linear Regression） 均方根误差（RMSE): %.8f；" % rmselist[1]
                + '\n' + "K Nearest Neighbours均方根误差(RMSE): %.8f；" % rmselist[2]+ '\n' + "差分整合移动平均自回归模型（ARIMA） 均方根误差（RMSE）:%.8f；" % rmselist[3] + \
            '\n'},inplace=True)

        
        # df.loc[df.index.values[0], 'predict_info'] = [
        #     "“Value”异常个数：%d；" % date_info[1] + '\n' + "“Date”异常个数：%d；" % date_info[3] + '\n' + "“Date”重复个数：%d；" %
        #     date_info[2] + '\n' + "缺失值总数为:%d；" % qs \
        #     + '\n' + "有效值总数为:%d；" % date_info[4] + '\n' + '每种方法均方差：' + '\n' + "移动平均（Moving Average） 均方根误差（RMSE): %.8f；" % rmselist[0] + '\n' + "线性回归（Linear Regression） 均方根误差（RMSE): %.8f；" % rmselist[1]
        #         + '\n' + "K Nearest Neighbours的均方根误差(RMSE): %.8f；" % rmselist[2]+ '\n' + "差分整合移动平均自回归模型（ARIMA） 均方根误差（RMSE）:%.8f；" % rmselist[3] + \
        #     '\n' + "长短期记忆人工神经网络(LSTM) 均方根误差(RMSE):%.8f。" % rmselist[4]]
        
        
    # df['predict']='未缺失'
    # row_indexer = df[df['value'].isnull()].index.tolist()
    # df.loc[row_indexer, 'predict'] = '预测'


    return df, fig