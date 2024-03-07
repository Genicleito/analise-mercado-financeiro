import pandas as pd
import numpy as np
import scipy
import scipy.stats
import datetime
from tqdm import tqdm
import plotly.graph_objects as go

TICKERS = list(set([
    'CEAB3', 'OIBR3', 'EMBR3', 'VALE3', 'GOLL4', 'COGN3', 'IRBR3', 'ABEV3', 'BBDC4', 'VULC3', 'SUZB3', 'AZUL4', 'QUAL3', 'SEER3',
    'BMGB4', 'ECOR3', 'TOTS3', 'ITUB4', 'LREN3', 'GGBR4', 'USIM5', 'MRFG3', 'RENT3', 'MOVI3', 'VIVA3', 'ARZZ3', 'ETER3',
    'BRKM5', 'PFRM3', 'SOMA3', 'ABCB4', 'AMAR3', 'ANIM3', 'BPAN4', 'BRPR3', 'PETR4', 'SAPR3', 'MEAL3', 'TEND3', 'CIEL3', 'MILS3',
    'CCRO3', 'BEEF3', 'MGLU3', 'BBAS3', 'WEGE3', 'CYRE3', 'JHSF3', 'KLBN11', 'SHOW3', 'MRVE3', 'CSAN3', 'NTCO3', 'MDNE3',
    'SAPR11', 'JBSS3', 'BRFS3', 'CSNA3', 'ELET3', 'CMIG4', 'PDGR3', 'LPSB3', 'PRNR3', 'EZTC3', 'ENAT3', 'DMVF3', 'GUAR3',
    'SBSP3', 'RANI3', 'LWSA3', 'SAPR4', 'CAML3', 'GRND3', 'AGRO3', 'CRFB3', 'LAVV3', 'PGMN3', 'SMTO3', 'MYPK3', 'POMO4', 'STBP3', 'PETZ3',
    'ITSA4', 'PTBL3', 'ENJU3', 'AERI3', 'GMAT3', 'CRFB3', 'RAPT4', 'CXSE3', 'BHIA3', 'PETR3', 'ITUB3', 'OIBR4', 'BBSE3',
]))

CRYPTOS = list(set([
    'BTC', 'ETH', 'USDT', 'BNB', 'OL', 'XRP', 'USDC', 'ADA', 'DOGE', 'SHIB', 'AVAX', 'TRX', 'DOT', 'LINK', 'MATIC', 'TON', 'UNI', 'PIXEL',
    'BCH', 'LTC', 'DAI', 'APT', 'ATOM', 'FIL', 'OP', 'TAO', 'IMX', 'STX', 'XLM', 'HBAR', 'CRO', 'INJ', 'KAS', 'OKB', 'VET', 'MNT', 'PEPE',
    'LDO', 'XMR', 'TIA', 'RNDR', 'ARB', 'AR', 'BEAM', 'BONK', 'BSV', 'ALGO', 'MKR', 'SEI', 'FTM', 'SUI', 'RUNE', 'MEME', 'STRK', 'FLOW',
    'XEGLD', 'ORDI', 'WIF', 'AAVE', 'FET', 'SAND', 'QNT', 'FLR', 'AXS', 'HNT', 'TUSD', '1000SATS', 'MINA', 'XEC', 'BGB', 'XTZ', 'KCS', 'APE',
    'SNX', 'AXL', 'CHZ', 'LUNC', 'MANA', 'NEO', 'EOS', 'FLOKI', 'GALA', 'ETHDYDX', 'JASMY', 'AGIX', 'IOTA', 'CFX', 'ROSE', 'AKT', 'GNO',
    'PYTH', 'KAVA', 'WLD', 'WOO'
]))

DEFAULT_YAHOO_COLUMNS = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']

def now(as_date_text=False):
    if as_date_text:
        return datetime.datetime.now().strftime('%Y-%m-%d')
    return datetime.datetime.now()

def cox_stuart_test(X, p=0.05, trend_type='d', debug=True):
    """Função pararealizar o teste de hipótese de Cox-Stuart.

    Parameters:
        * `X` list: Vetor com os valores numéricos.
        * `p` float: Valor de probabilidade p para teste de hipótese. Default: `0.05`.
        * `trend_type` str: Tipo da tendência a ser calculada, sendo `'d'` *decreasing trend* e
            `'i'` *increasing trend*.
    """
    ts = now()
    x = X[:]
    if len(x) % 2 == 1:
        id_del = len(x) // 2
        x = np.delete(x, id_del)

    half = len(x) // 2

    x1 = x[np.arange(0, half, dtype=int)]
    x2 = x[np.arange(half, len(x), dtype=int)]

    n = np.sum((x2 - x1) != 0)
    t = np.sum(x1 < x2)

    if trend_type == 'i':
        p_value = 1 - scipy.stats.binom.cdf(t - 1, n, p)
        if debug: print(f"P = {p} | P-Value = {p_value}")
        if debug: print(f"\t{'Não há tendência sgnificativa de baixa!' if p_value >= p else 'Há tendência significativa de baixa!'}")
    else: # elif trend_type == "d"
        p_value = scipy.stats.binom.cdf(t, n, p)
        if debug: print(f"P = {p} | P-Value = {p_value}")
        if debug: print(f"\t{'Não há tendência sgnificativa de alta!' if p_value >= p else 'Há tendência significativa de alta!'}")
    
    if debug: print(f"Tempo de execução: {now() - ts}")
    return p_value, p_value < p

def min_values(np_arrays):
    r = np_arrays[0]
    for i in range(1, len(np_arrays), 1):
        r = np.where(r < np_arrays[i], r, np_arrays[i])
    return r

def get_market_data():
    def create_ema(df, periods=[8, 20, 72, 200]):
        dfs = []
        tmp = df.copy()
        if not isinstance(periods, list): periods = [periods]
        for code in tmp['ticker'].unique():
            tmp = tmp[tmp['ticker'] == code].sort_values(by=['date'])
            for p in periods:
                # call ema function
                # tmp[f'close_ema{p}'] = tmp['close'].ewm(span=p, min_periods=p, adjust=False).mean().round(2)
                tmp[f'close_ema{p}'] = tmp['close'].rolling(window=p).mean().round(2)
                if p == 20:
                    # EMA volume 20 periods
                    # tmp[f'volume_ema{p}'] = tmp['volume'].ewm(span=p, min_periods=p, adjust=False).mean().round(0).astype(int, errors='ignore')
                    tmp[f'volume_ema{p}'] = tmp['volume'].rolling(window=p).mean().round(0).astype(int, errors='ignore')
            dfs.append(tmp)
        return pd.concat(dfs, ignore_index=True)

    def ema(serie, period):
        return serie.ewm(span=period, min_periods=period, adjust=False).mean()

    def macd(fast_ma, slow_ma):
        return fast_ma - slow_ma

    def flag_volume(df):
        return df.assign(
            ind_volume=np.where(
                df['volume'].notna() & (df['volume'] < df['volume_ema20']),
                -1, np.where(df['volume'].notna() & (df['volume'] > df['volume_ema20']), 1, 0)
            )
        )

    def candle_crossing_ema(df, period):
        return np.where(((df['close'] > df[f'close_ema{period}']) & (df['close'].shift(1) <= df[f'close_ema{period}'].shift(1))), 1, 0)

    def create_yahoo_download_query(code, period1=946695600, debug=False, crypto=False):
        now = datetime.datetime.today()
        _YAHOO_API_URL = 'https://query1.finance.yahoo.com/v7/finance/download/'

        period2 = int(datetime.datetime(year=now.year, month=now.month, day=now.day, hour=max(now.hour - 1, 0), minute=0, second=0, microsecond=0).timestamp())
        # if _DEBUG: print(f'Period1: {datetime.datetime.fromtimestamp(period1)} | Period2: {datetime.datetime.fromtimestamp(period2)}')
        url = f'{yahoo_api_url}{code}{".SA" if not crypto else "-USD"}?period1={period1}&period2={period2}&interval=1d&events=history'
        if debug: print(f"Query: {url}")
        return url

    def get_yahoo_finance(tickers):
        dfs = []
        if not isinstance(tickers, list): tickers = [tickers]
        success = []
        errors = []
        for ticker in tickers:
            url = create_yahoo_download_query(code=ticker, crypto=ticker in CRYPTOS)
            try:
                tmp = pd.read_csv(url).assign(**{
                    'ticker': ticker if not ticker in CRYPTOS else f"{ticker}USD"
                })
                tmp = tmp.rename(columns={x: x.lower().replace(' ', '_') for x in tmp.columns})[DEFAULT_YAHOO_COLUMNS]
                tmp = tmp[tmp != 'null'].dropna()
                tmp = tmp.assign(
                    pct=(tmp['close'] - tmp['open']) / tmp['open'] * 100,
                    pct_ant=(tmp['close'] - tmp['close'].shift(1)) / tmp['close'].shift(1) * 100
                )
                for col in DEFAULT_YAHOO_COLUMNS:
                    tmp[col] = tmp[col].astype(float, errors='ignore')
                for col in tmp.columns:
                    tmp[col] = tmp[col].astype(float, errors='ignore')
                dfs.append(tmp)
                del(tmp)
                success.append(ticker)
            except Exception as exp:
                errors.append(ticker)
                print(f'{now()} [{ticker}] Error: {exp}')
        return pd.concat(dfs, ignore_index=True).sort_values('date') if dfs else pd.DataFrame()
    
    prop_return_risk = 2.5
    dfs = []
    for ticker in tqdm(set(TICKERS + CRYPTOS)):
        tmp = get_yahoo_finance(tickers=ticker)
        if tmp.shape[0] > 1:
            tmp = tmp.rename(columns={'code': 'ticker'})
            tmp = create_ema(tmp.drop_duplicates(['date', 'ticker']))
            tmp = flag_volume(tmp)
            tmp = tmp.assign(**{
                'ma26': tmp['close'].rolling(window=26).mean(),
                'ma12': tmp['close'].rolling(window=12).mean()
            })
            tmp['macd'] = macd(fast_ma=tmp['ma26'], slow_ma=tmp['ma12'])
            tmp['macd_signal'] = ema(serie=tmp['macd'], period=9)
            # tmp = get_signals(tmp)
            tmp = tmp.assign(**{
                'candle_crossing_ema20': candle_crossing_ema(tmp, period=20),
                'crossing_8ema_x_20ema': np.where((tmp['close_ema8'] >= tmp['close_ema20']) & (tmp['close_ema8'].shift(1) < tmp['close_ema20'].shift(1)), 1, 0),
                'crossing_8ema_x_72ema': np.where((tmp['close_ema8'] >= tmp['close_ema72']) & (tmp['close_ema8'].shift(1) < tmp['close_ema72'].shift(1)), 1, 0),
                'crossing_20ema_x_72ema': np.where((tmp['close_ema20'] >= tmp['close_ema72']) & (tmp['close_ema20'].shift(1) < tmp['close_ema72'].shift(1)), 1, 0),
                'trend_tomorrow_%': (tmp['close'].shift(-1) - tmp['close']) / (tmp['close'].shift(-1)) * 100,
                'gain_8_%': ((tmp['close'] - tmp['close'].shift(-8)) / tmp['close']) * 100,
                'gain_20_%': ((tmp['close'] - tmp['close'].shift(-20)) / tmp['close']) * 100,
                'gain_72_%': ((tmp['close'] - tmp['close'].shift(-72)) / tmp['close']) * 100,
                'stop_loss': min_values([tmp['low'], tmp['low'].shift(1), tmp['low'].shift(2)]),
                'stop_gain_suggested': prop_return_risk * (tmp['close'] -  min_values([tmp['low'], tmp['low'].shift(1), tmp['low'].shift(2)])), # 2 * risk
            })
            dfs.append(tmp)
            del(tmp)
    # Une os DataFrames
    df = pd.concat(dfs, ignore_index=True)

    # Cria colunas com base nas informações dos candles
    candle_length = df['high'] - df['low']
    df = df.assign(
        **{
            'volume_to_average': (df['volume'] / df['volume_ema20']),
            'macd_to_average': (df['macd'] / df['macd_signal']),
            'candle_lose': (df['high'] - df['close']) / candle_length * 100,
            'candle_gain': (df['close'] - df['low']) / candle_length * 100,
            'candle_length': candle_length,
            'candle_body': (df['close'] - df['open']) / candle_length * 100,
            'candle_variation_%': ((df['close'] - df['open']) / df['close']) * 100,
            'lower_shadow_%': ((np.where(df['open'] < df['close'], df['open'], df['close']) - df['low']) / candle_length) * 100,
            'upper_shadow_%': ((df['high'] - np.where(df['open'] > df['close'], df['open'], df['close'])) / candle_length) * 100,
            'ema8_over_20': df['close_ema8'] - df['close_ema20'],
            'ema8_over_72': df['close_ema8'] - df['close_ema72'],
            'ema20_over_72': df['close_ema20'] - df['close_ema72'],
        }
    )
    # Cria coluna(s) que depende(m) das criadas anteriormente
    return df.assign(**{
        'tend_alta_medias': (
            ((df['ema8_over_20'] > 0) & (df['ema8_over_20'].shift(1) <= 0))
                | ((df['ema8_over_72'] > 0) & (df['ema8_over_72'].shift(1) <= 0))
                | ((df['ema20_over_72'] > 0) & (df['ema20_over_72'].shift(1) <= 0))
        )
    }).sort_values('date', ascending=False)

def holt_winters(df_ticker, periods_forecast=20, seasonal_periods=25, seasonal='mul', prod=False, debug=True): # mul or add
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    X = df_ticker['close']
    if not prod:
        X_train = X[:len(X) - periods_forecast]
        X_test = X[-periods_forecast: ]

        if debug: print(f"X train: {len(X_train)} | X test: {len(X_test)}")
    else:
        X_train = X
        X_test = None
    
    # Model training (Multiplicativo)
    model = ExponentialSmoothing(
        X_train,
        trend=seasonal, seasonal=seasonal,
        seasonal_periods=seasonal_periods
    ).fit()

    # Prediction
    Y = model.forecast(periods_forecast)

    return X_train, X_test, Y

def plot_serie(df_ticker, column='close'):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=df_ticker[column],
            mode='lines',
            name='Preço de Fechamento',
            line = dict(width=2.5)
        )
    )

    fig.add_trace(
        go.Scatter(
            y=df_ticker['Média Móvel (8p)'],
            mode='lines',
            name='Média Móvel (8p)',
            line = dict(color='#cc66cc', width=1.25) # , dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(
            y=df_ticker['Média Móvel (20p)'],
            mode='lines',
            name='Média Móvel (20p)',
            line = dict(color='#000000', width=1.25) # , dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(
            y=df_ticker['Média Móvel (72p)'],
            mode='lines',
            name='Média Móvel (72p)',
            line = dict(color='#99994d', width=1.25) # , dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(
            y=df_ticker['Média Móvel (200p)'],
            mode='lines',
            name='Média Móvel (200p)',
            line = dict(color='#5c5c8a', width=1.25) # , dash='dash')
        )
    )

    fig.update_layout(
        title=f"Últimos {df_ticker.shape[0]} pregões na Bolsa de Valores da ação {df_ticker['ticker'].iloc[0]} [Último registro: {df_ticker['date'].max()}]",
        xaxis_title="Períodos (dias)",
        yaxis_title="Preço de fechamento (R$)",
        legend_title="Série e médias móveis",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        #     color="RebeccaPurple"
        # )
    )

    annotations = []
    annotations.append(
        dict(
            xref='paper', yref='paper', x=0.9, y=-0.1,
            xanchor='center', yanchor='top', text=f"Último registro: {df_ticker['date'].max()}",
            # font=dict(
            #     family='Arial',
            #     size=12,
            #     color='rgb(150,150,150)'
            # ),
            showarrow=False
        )
    )

    fig.update_layout(annotations=annotations)
    return fig

def plot_model_results(df_ticker, X_train, X_test, Y, method_name):
    X = df_ticker['close']
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=X_train.tolist() + [None] * (len(X) - len(X_train)),
            mode='lines',
            name='Valor de Treinamento',
            line = dict(width=1.25)
        )
    )

    fig.add_trace(
        go.Scatter(
            y=[None] * len(X_train) + X_test.tolist(),
            mode='lines',
            name='Valor Real',
            line = dict(color='#cc66cc', width=1.25) # , dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(
            y=[None] * len(X_train) + Y.tolist(),
            mode='lines',
            name='Valor Predito',
            line = dict(color='#000000', width=1.25) # , dash='dash')
        )
    )

    fig.update_layout(
        title=f"Predição dos valores de fechamento de {df_ticker['ticker'].iloc[0]} [{method_name}]",
        xaxis_title="Períodos (dias)",
        yaxis_title="Preço (R$)",
        legend_title="",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        #     color="RebeccaPurple"
        # )
    )

    annotations = []
    annotations.append(
        dict(
            xref='paper', yref='paper', x=0.9, y=-0.1,
            xanchor='center', yanchor='top', text=f"Último registro: {df_ticker['date'].max()}",
            # font=dict(
            #     family='Arial',
            #     size=12,
            #     color='rgb(150,150,150)'
            # ),
            showarrow=False
        )
    )

    fig.update_layout(annotations=annotations)
    return fig
