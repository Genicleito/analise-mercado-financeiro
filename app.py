import streamlit as st
import os

@st.cache_data
def install_requirements():
    os.system("pip install -r requirements.txt")
install_requirements()

import pandas as pd
import requests
import plotly.graph_objects as go
import datetime
from lib import models

lib_path = models.LIB_PATH
url_lib = models.URL_LIB
with open(lib_path, 'wb+') as f:
    f.write(requests.get(url_lib).text.encode('utf-8'))

from lib import technical_analysis

@st.cache_resource
def load_data():
    # return technical_analysis.daily_analysis_yfinance()
    return pd.read_csv(models.READ_MARKET_DATA_PATH), None

df = pd.DataFrame()
with st.status('Loading data...'):
    st.write(f'_**{datetime.datetime.now()}** Lendo dados... Aguarde alguns segundos..._')
    df, _ = load_data()
    st.write(f'{datetime.datetime.now()} Dados lidos com sucesso: {df.shape[0]} linhas')


if df.shape[0] > 0:
    option = st.selectbox(
        'Selecione um código de ativo:',
        options=sorted(df['ticker'].unique())
    )

data = df[(df['ticker'] == 'PETR4')]  # & (df['date'].dt.date >= (datetime.datetime.today() - datetime.timedelta(days=20)).date())]
fig = go.Figure(
    data=[
        go.Candlestick(
            x=data['date'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close']
        )
    ]
)

st.plotly_chart(fig, use_container_width=True)


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

