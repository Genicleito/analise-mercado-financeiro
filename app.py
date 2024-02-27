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
    # return pd.read_csv(models.READ_MARKET_DATA_PATH)
    try:
        df = utils.get_market_data()
    except:
        print(f"[load_data()] Dados não puderam ser baixados!. \n\tLendo arquivos armazenados em:\n\t{models.READ_MARKET_DATA_PATH}")
        df = pd.read_csv(models.READ_MARKET_DATA_PATH)
    
    if df.shape[0] == 0:
        print(f'Dados vazios, lendo dados salvos no repositório.')
        df = pd.read_csv(models.READ_MARKET_DATA_PATH)
    return df
    

df = pd.DataFrame()
with st.status('Loading data...'):
    ts = datetime.datetime.now()
    st.write(f"_{ts.strftime('%Y-%m-%d %H:%M:%S')} Lendo dados... Aguarde alguns instantes..._")
    df = load_data()
    st.write(f"_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Dados lidos com sucesso em {datetime.datetime.now() - ts} [{len(df['ticker'].unique())} ativos]_")

if df.shape[0] > 0:
    # Criar duas colunas: uma para o ticker e outra para periodos
    col_ticker, col_periods = st.columns(2, gap='medium')
    with col_ticker:
        ticker_sb = st.selectbox(
            '*Selecione um código de ativo:*',
            options=[''] + sorted(df['ticker'].unique())
        )
        # Cria um novo DataFrame apenas com o ativo selecionado
        df_ticker = df.query(f"ticker == '{ticker_sb}'")
    
    with col_periods:
        periods_sb = st.selectbox(
            '*Selecione o período para teste de hipótese:*',
            options=[21, 9, 45, 72, 200]
        )

    if ticker_sb:
        st.write(f"Dados da *{ticker_sb}* atualizados até `{df[df['ticker'] == ticker_sb]['date'].max()}`")

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

st.write(f"### Gráfico com visualização da série do ativo no TradingView")
components.html(
    html=models.get_widget_trading_view(ticker=ticker_sb if ticker_sb else 'IBOV'),
    height=400,
    # width=900
)

if ticker_sb:
    st.markdown(f"# Teste de Tendência")
    periods = periods_sb if periods_sb else models.PERIODS_H_TEST
    r = run_cox_stuart_test(df, ticker=ticker_sb, periods=periods)

    if r[0][1]:
        st.write(f"* **Há tendência significativa de alta 📈:** \n\t * **p-value** = {r[0][0]} \n\t * **periods** = {periods}")
    else:
        st.write(f"* **Não há tendência significativa de alta 😐:** \n\t * **p-value** = {r[0][0]} \n\t * **periods** = {periods}")

    if r[1][1]:
        st.write(f"* **Há tendência significativa de baixa 📉:** \n\t * **p-value** = {r[1][0]} \n\t * **periods** = {periods}")
    else:
        st.write(f"* **Não há tendência significativa de baixa 😐:** \n\t * **p-value** = {r[1][0]} \n\t * **periods** = {periods}!")

    st.dataframe(
        df_ticker[
            df_ticker['date'] == df_ticker['date'].max()
        ].reset_index(drop=True).round(4).T.rename(columns={0: 'info'}),
        width=600
    )

    st.write(f"#### Ações com tendência de alta por cruzamento de médias na data mais recente [`{df['date'].max()}`]")
    st.dataframe(
        df[
            (df['tend_alta_medias'] == True) & (df['date'] == df['date'].max())
        ].sort_values('date', ascending=False).reset_index(drop=False)
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

