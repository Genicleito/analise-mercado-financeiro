import streamlit as st
import streamlit.components.v1 as components
import os

@st.cache_data
def install_requirements():
    os.system("pip install -r requirements.txt")
install_requirements()

import pandas as pd
import requests
import plotly.graph_objects as go
import datetime
from lib import (
    models, utils
)

# lib_path = models.LIB_PATH
# url_lib = models.URL_LIB
# with open(lib_path, 'wb+') as f:
#     f.write(requests.get(url_lib).text.encode('utf-8'))

# # Importa biblioteca adicional obtida do repositório do github
# from lib import technical_analysis

def run_cox_stuart_test(df, ticker, periods=None): # GOLL4, 21
    # Prepare data
    data = df.query(f"ticker == '{ticker}'").sort_values('date').copy()

    # Unidimensional
    X = data['close'].to_numpy()
    # Considera todos os dados caso nenhum período tenha sido selecionado
    if not periods: periods = len(X)
    X = X[-periods:]

    print(f"ticker = {ticker}\nperiods = {data['date'].iloc[-periods]} <-> {data['date'].iloc[-1]}")
    p_value_alta, tend_alta = utils.cox_stuart_test(X, p=models.P, trend_type='i')
    p_value_baixa, tend_baixa = utils.cox_stuart_test(X, p=models.P, trend_type='d')

    return [(p_value_alta, tend_alta), (p_value_baixa, tend_baixa)]

@st.cache_resource
def load_data():
    # return technical_analysis.daily_analysis_yfinance()
    return pd.read_csv(models.READ_MARKET_DATA_PATH)

df = pd.DataFrame()
with st.status('Loading data...'):
    st.write(f'_**{datetime.datetime.now()}** Lendo dados... Aguarde alguns segundos..._')
    df = load_data()
    st.write(f'{datetime.datetime.now()} Dados lidos com sucesso: {df.shape[0]} linhas')

if df.shape[0] > 0:
    # Criar duas colunas: uma para o ticker e outra para periodos
    col_ticker, col_periods = st.columns(2, gap='medium')
    with col_ticker:
        ticker_sb = st.selectbox(
            '#### Selecione um código de ativo:',
            options=[''] + sorted(df['ticker'].unique())
        )
    
    with col_periods:
        periods_sb = st.selectbox(
            '#### Selecione o período para filtrar dados no teste de hipótese:',
            options=[21, 9, 72, 200]
        )

    if ticker_sb:
        st.write(f"Dados para {ticker_sb} atualizados até `{df[df['ticker'] == ticker_sb]['date'].max()}`")

# if ticker_sb:
#     data = df[(df['ticker'] == ticker_sb)]  # & (df['date'].dt.date >= (datetime.datetime.today() - datetime.timedelta(days=20)).date())]
#     fig = go.Figure(
#         data=[
#             go.Candlestick(
#                 x=data['date'],
#                 open=data['open'],
#                 high=data['high'],
#                 low=data['low'],
#                 close=data['close']
#             )
#         ]
#     )

#     st.plotly_chart(fig, use_container_width=True)

if ticker_sb:
    st.markdown(f"# Teste de Tendência")
    r = run_cox_stuart_test(df, ticker=ticker_sb, periods=periods_sb is periods_sb else models.PERIODS_H_TEST)

    if r[0][1]:
        st.write(f"Há tendência sgnificativa de alta `[p-value = {r[0][0]} | periods = {models.PERIODS_H_TEST}]`!")
    else:
        st.write(f"Não há tendência sgnificativa de alta `[p-value = {r[0][0]} | periods = {models.PERIODS_H_TEST}]`!")

    if r[1][1]:
        st.write(f"Há tendência sgnificativa de baixa `[p-value = {r[1][0]} | periods = {models.PERIODS_H_TEST}]`!")
    else:
        st.write(f"Não há tendência sgnificativa de baixa `[p-value = {r[1][0]} | periods = {models.PERIODS_H_TEST}]`!")

components.html(
    html=models.get_widget_trading_view(ticker=ticker_sb if ticker_sb else 'IBOV'),
    height=900,
    width=1200
)

################# Adicionando textos e anotações ##############
# import plotly.graph_objects as go
# import pandas as pd

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

# fig = go.Figure(data=[go.Candlestick(x=df['Date'],
#                 open=df['AAPL.Open'], high=df['AAPL.High'],
#                 low=df['AAPL.Low'], close=df['AAPL.Close'])
#                       ])

# fig.update_layout(
#     title='The Great Recession',
#     yaxis_title='AAPL Stock',
#     shapes = [dict(
#         x0='2016-12-09', x1='2016-12-09', y0=0, y1=1, xref='x', yref='paper',
#         line_width=2)],
#     annotations=[dict(
#         x='2016-12-09', y=0.05, xref='x', yref='paper',
#         showarrow=False, xanchor='left', text='Increase Period Begins')]
# )

# fig.show()

