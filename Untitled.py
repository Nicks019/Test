import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.stats.diagnostic as diag
import statsmodels.api as sm

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Stock Forecast App')

def Category_19():
    df = pd.read_csv(r"Historical Product Demand.csv",parse_dates=['Date'])
    index = df[ df['Order_Demand'] <1000 ].index
    df.drop(index,inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    index1 = df[df['Year'] == 2011 ].index
    df.drop(index1,inplace=True)
    index2 = df[df['Year'] == 2017].index
    df.drop(index2,inplace=True)
    df.drop(['Year','Month'],axis=1,inplace=True)
    df.dropna(axis=0, inplace=True)
    q1=df['Order_Demand'].quantile(0.25)
    q2=df['Order_Demand'].quantile(0.50)
    q3=df['Order_Demand'].quantile(0.75)
    iqr=q3-q1
    upper_limit=q3+1.5*iqr
    lower_limit=q1-1.5*iqr
    upper_limit,lower_limit
    def limit_imputer(value):
        if value > upper_limit:
            return upper_limit
        if value < lower_limit:
            a=a+1
            return lower_limit
        else:
            return value
    df['Order_Demand']=df['Order_Demand'].apply(limit_imputer)


    #Product = ['Category_19', 'Category_06', 'Category_05','Category_07','Category_28']
    #Selected_Product = st.selectbox('Select dataset for prediction', Product)
    #N_Month = int(st.text_input(" Input Forecast Months ", 24))

# st.button('Category_19')


    li = ['Category_019','Category_006','Category_028','Category_005','Category_007']
    df19 = df[df.Product_Category==li[0]]
    df19= df19.groupby('Date')['Order_Demand'].count().reset_index()
    df19 = df19.set_index(['Date'])
    df19= df19['Order_Demand'].resample('MS').mean()
    df19 = df19.fillna(df19.bfill())
    df_19=df19.to_frame()
    decomposition = sm.tsa.seasonal_decompose(df_19, model='multiplicative')
    model=sm.tsa.statespace.SARIMAX(df19,order=(1,1,1),seasonal_order=(1,1,0,12))
    results=model.fit()
    pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=True)
    pred_ci = pred.conf_int()
    pred_uc = results.get_forecast(steps = N_Month)
    pred_ci = pred_uc.conf_int()
    ax = df19.plot(label='observed', figsize=(16, 8))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Order_Demand')
    plt.show()
    plt.legend()
    st.pyplot()
    FORECAST_19 = results.forecast(steps = N_Month)
    FORECAST_19=FORECAST_19.to_frame()
    inventory_management_list_19 = FORECAST_19['predicted_mean'].tolist()
    stock=0
    refill_list=[]
    balanced_stock=[]
    order_placed=[]
    extra_order_for_refill=[]
    for x in inventory_management_list_19:
        if stock<=(x*1.2):
            Extra_order=(x*1.2)-stock
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x#balancedStock
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
        else:
            Extra_order=0
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
    df = pd.DataFrame(list(zip(inventory_management_list_19,extra_order_for_refill,refill_list,order_placed,balanced_stock)), columns =['order_demand','Refill_0rder','refill_list','order','balanced'],index=FORECAST_19.index)
    st.write(df)
    return(df,ax,FORECAST_19)
def Category_06():
    df = pd.read_csv(r"Historical Product Demand.csv",parse_dates=['Date'])
    index = df[ df['Order_Demand'] <1000 ].index
    df.drop(index,inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    index1 = df[df['Year'] == 2011 ].index
    df.drop(index1,inplace=True)
    index2 = df[df['Year'] == 2017].index
    df.drop(index2,inplace=True)
    df.drop(['Year','Month'],axis=1,inplace=True)
    df.dropna(axis=0, inplace=True)
    q1=df['Order_Demand'].quantile(0.25)
    q2=df['Order_Demand'].quantile(0.50)
    q3=df['Order_Demand'].quantile(0.75)
    iqr=q3-q1
    upper_limit=q3+1.5*iqr
    lower_limit=q1-1.5*iqr
    upper_limit,lower_limit
    def limit_imputer(value):
        if value > upper_limit:
            return upper_limit
        if value < lower_limit:
            a=a+1
            return lower_limit
        else:
            return value
    df['Order_Demand']=df['Order_Demand'].apply(limit_imputer)


    #Product = ['Category_19', 'Category_06', 'Category_05','Category_07','Category_28']
    #Selected_Product = st.selectbox('Select dataset for prediction', Product)
    #N_Month = int(st.text_input(" Input Forecast Months ", 24))

# st.button('Category_19')


    li = ['Category_019','Category_006','Category_028','Category_005','Category_007']
    df06 = df[df.Product_Category==li[1]]
    df06= df06.groupby('Date')['Order_Demand'].count().reset_index()
    df06 = df06.set_index(['Date'])
    df06= df06['Order_Demand'].resample('MS').mean()
    df06 = df06.fillna(df06.bfill())
    df_06=df06.to_frame()
    decomposition = sm.tsa.seasonal_decompose(df_06, model='multiplicative')
    model=sm.tsa.statespace.SARIMAX(df06,order=(1,1,1),seasonal_order=(1,1,0,12))
    results=model.fit()
    pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=True)
    pred_ci = pred.conf_int()
    pred_uc = results.get_forecast(steps = N_Month)
    pred_ci = pred_uc.conf_int()
    ax = df06.plot(label='observed', figsize=(16, 8))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Order_Demand')
    plt.show()
    plt.legend()
    st.pyplot()
    FORECAST_06 = results.forecast(steps = N_Month)
    FORECAST_06=FORECAST_06.to_frame()
    inventory_management_list_06 = FORECAST_06['predicted_mean'].tolist()
    stock=0
    refill_list=[]
    balanced_stock=[]
    order_placed=[]
    extra_order_for_refill=[]
    for x in inventory_management_list_06:
        if stock<=(x*1.2):
            Extra_order=(x*1.2)-stock
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x#balancedStock
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
        else:
            Extra_order=0
            stock=stock+Extra_order
            refill_list.append(stock)
            stock=stock-x
            balanced_stock.append(stock)
            order_placed.append(x)
            extra_order_for_refill.append(Extra_order)
    df = pd.DataFrame(list(zip(inventory_management_list_06,extra_order_for_refill,refill_list,order_placed,balanced_stock)), columns =['order_demand','Refill_0rder','refill_list','order','balanced'],index=FORECAST_06.index)
    st.write(df)
    return(df,ax,FORECAST_06)
if st.button('Category_19'):
    N_Month = int(st.text_input(" Input Forecast Months ", 24))
    Category_19()
elif st.button('Category_06'):
    N_Month = int(st.text_input(" Input Forecast Months ", 24))
    Category_06()
else:
    print('')
    
    
    
    
    

