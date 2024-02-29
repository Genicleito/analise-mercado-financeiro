import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

setattr(pd, "Int64Index", pd.Index)
setattr(pd, "Float64Index", pd.Index)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

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
    online_data = False
    try:
        df = utils.get_market_data()
        online_data = True
    except:
        print(f"[load_data()] Dados não puderam ser baixados!. \n\tLendo arquivos armazenados em:\n\t{models.READ_MARKET_DATA_PATH}")
        df = pd.read_csv(models.READ_MARKET_DATA_PATH)
    
    if df.shape[0] == 0:
        print(f'Dados vazios, lendo dados salvos no repositório.')
        df = pd.read_csv(models.READ_MARKET_DATA_PATH)
    return df, online_data
    

df = pd.DataFrame()
with st.status('Loading data...'):
    ts = datetime.datetime.now()
    st.write(f"_{ts.strftime('%Y-%m-%d %H:%M:%S')} Lendo dados... Aguarde alguns instantes..._")
    df, online_data = load_data()
    df = df.rename(columns={
        'close_ema8': 'Média Móvel (8p)',
        'close_ema20': 'Média Móvel (20p)',
        'close_ema72': 'Média Móvel (72p)',
        'close_ema200': 'Média Móvel (200p)',
    })# .query("date <= '2024-02-26'")
    st.write(f"_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Dados lidos com sucesso em {datetime.datetime.now() - ts} [{len(df['ticker'].unique())} ativos]_")

if df.shape[0] > 0:
    st.write(f"Dados{' [*offline*]' if not online_data else ''} atualizados até `{df['date'].max()}`")

    st.write(f"# Input de informações")
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
        st.write(f"#### Dados de *{ticker_sb}* na data *{df_ticker['date'].max()}*:")
        st.dataframe(
            df_ticker[
                df_ticker['date'] == df_ticker['date'].max()
            ].reset_index(drop=True).round(4).T.rename(columns={0: 'info'}),
            width=400
        )

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
    st.markdown(f"# Teste de Tendência (*Cox-Stuart Test*)")
    periods = periods_sb if periods_sb else models.PERIODS_H_TEST
    r = run_cox_stuart_test(df, ticker=ticker_sb, periods=periods)

    if r[0][1]:
        st.write(f"* **Há tendência significativa de alta 📈:** \n\t * **p-value** = {r[0][0]} \n\t * **periods** = {periods}")
    else:
        st.write(f"* **Não há tendência significativa de alta 😐:** \n\t * **p-value** = {r[0][0]} \n\t * **periods** = {periods}")

    if r[1][1]:
        st.write(f"* **Há tendência significativa de baixa 📉:** \n\t * **p-value** = {r[1][0]} \n\t * **periods** = {periods}")
    else:
        st.write(f"* **Não há tendência significativa de baixa 😐:** \n\t * **p-value** = {r[1][0]} \n\t * **periods** = {periods}")

    st.write(f"### Ações com tendência de alta por cruzamento de médias na data mais recente [`{df['date'].max()}`]")
    st.dataframe(
        df[
            (df['tend_alta_medias'] == True) & (df['date'] == df['date'].max())
        ].drop(
            ['buy', 'sell', 'tend_alta_medias'], axis=1, errors='ignore'
        ).sort_values('date', ascending=False).reset_index(drop=True),
        width=900
    )

    # DataFrame para predição
    df_pred = df_ticker.iloc[-models.PERIODS_ANT_PREDICTION:]

    st.markdown(f"# Visualização e Análise da Série")
    fig = utils.plot_serie(df_pred)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"# Modelo de predição `Holt-Winters`")
    seasonal = 'mul' # mul or add
    X_train, X_test, Y = utils.holt_winters(df_pred, periods_forecast=models.PERIODS_FORECAST, seasonal_periods=25, seasonal=seasonal, debug=True)

    st.markdown(f"### Resultados do modelo")
    fig_pred = utils.plot_model_results(df_pred, X_train, X_test, Y, seasonal)
    st.plotly_chart(fig_pred, use_container_width=True)
    
    st.markdown(f"### Validação do Modelo")
    d = pd.DataFrame({
        'Medida (Multiplicativo)': ['MSE', 'RMSE', 'MAE', 'MAPE'],
        'Valor': [mse(X_test, Y), np.sqrt(mse(X_test, Y)), mae(X_test, Y), mape(X_test, Y)],
    })

    # DataFrame com as medidas de validação
    st.dataframe(d, use_container_width=True, hide_index=True) #, width=400)

    # DataFrame com valores reais e preditos
    st.dataframe(
        df_pred[['date', 'close']].sort_values('date', ascending=False).rename(
            columns={'close': 'Valor Real', 'date': 'Data do Pregão'}
        ).loc[X_test.index].assign(**{
            'Valor Predito': Y.to_numpy(),
            'MAE': (X_test - Y.to_numpy()).abs()
        }),
        use_container_width=True,
        hide_index=True
    )

    pregoes_predict = 7
    st.markdown(f"## Predição para o próximo pregão")
    _, _, pred = utils.holt_winters(df_pred, periods_forecast=7, prod=True, debug=True)

    st.metric(label="Preço Predito:", value=f"R$ {round(pred.iloc[0], 2)}", delta=f"{round((pred.iloc[0] - df_pred['close'].iloc[-1]) / df_pred['close'].iloc[-1] * 100, 2)}%")
