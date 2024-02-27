import pandas as pd
import numpy as np
import scipy
import scipy.stats
import datetime
from tqdm import tqdm

TICKERS = set([
    'CEAB3', 'OIBR3', 'EMBR3', 'VALE3', 'GOLL4', 'COGN3', 'IRBR3', 'ABEV3', 'BBDC4', 'VULC3', 'SUZB3', 'AZUL4', 'QUAL3', 'SEER3',
    'BMGB4', 'ECOR3', 'TOTS3', 'ITUB4', 'LREN3', 'GGBR4', 'USIM5', 'MRFG3', 'RENT3', 'MOVI3', 'VIVA3', 'ARZZ3', 'ETER3',
    'BRKM5', 'PFRM3', 'SOMA3', 'ABCB4', 'AMAR3', 'ANIM3', 'BPAN4', 'BRPR3', 'PETR4', 'SAPR3', 'MEAL3', 'TEND3', 'CIEL3', 'MILS3',
    'CCRO3', 'BEEF3', 'MGLU3', 'BBAS3', 'WEGE3', 'CYRE3', 'JHSF3', 'KLBN11', 'SHOW3', 'MRVE3', 'CSAN3', 'NTCO3', 'MDNE3',
    'SAPR11', 'JBSS3', 'BRFS3', 'CSNA3', 'ELET3', 'CMIG4', 'PDGR3', 'LPSB3', 'PRNR3', 'EZTC3', 'ENAT3', 'DMVF3', 'GUAR3',
    'SBSP3', 'RANI3', 'LWSA3', 'SAPR4', 'CAML3', 'GRND3', 'AGRO3', 'CRFB3', 'LAVV3', 'PGMN3', 'SMTO3', 'MYPK3', 'POMO4', 'STBP3', 'PETZ3',
    'ITSA4', 'PTBL3', 'ENJU3', 'AERI3', 'GMAT3', 'CRFB3', 'RAPT4', 'CXSE3', 'BHIA3', 'PETR3', 'ITUB3', 'OIBR4',
])

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

def get_market_data():
    def create_ema(df, periods=[8, 20, 72, 200]):
        dfs = []
        tmp = df.copy()
        if not isinstance(periods, list): periods = [periods]
        for code in tmp['ticker'].unique():
            tmp = tmp[tmp['ticker'] == code].sort_values(by=['date'])
            for p in periods:
                # call ema function
                tmp[f'close_ema{p}'] = tmp['close'].ewm(span=p, min_periods=p, adjust=False).mean().round(2)
                if p == 20:
                    # call ema function
                    tmp[f'volume_ema{p}'] = tmp['volume'].ewm(span=p, min_periods=p, adjust=False).mean().round(0).astype(int, errors='ignore')
            dfs.append(tmp)
        return pd.concat(dfs, ignore_index=True)

    def ema(serie, period):
        return serie.ewm(span=period, min_periods=period, adjust=False).mean().round(2)

    def macd(fast_ma, slow_ma, decimal=2):
        return (fast_ma - slow_ma).round(decimal)

    def flag_volume(df):
        return df.assign(
            ind_volume=np.where(
                df['volume'].notna() & (df['volume'] < df['volume_ema20']),
                -1, np.where(df['volume'].notna() & (df['volume'] > df['volume_ema20']), 1, 0)
            )
        )

    def candle_crossing_ema(df, period):
        return np.where(((df['close'] > df[f'close_ema{period}']) & (df['close'].shift(1) <= df[f'close_ema{period}'].shift(1))), 1, 0)

    def create_yahoo_download_query(code, period1=946695600, debug=False):
        now = datetime.datetime.today()
        _YAHOO_API_URL = 'https://query1.finance.yahoo.com/v7/finance/download/'

        period2 = int(datetime.datetime(year=now.year, month=now.month, day=now.day, hour=max(now.hour - 1, 0), minute=0, second=0, microsecond=0).timestamp())
        # if _DEBUG: print(f'Period1: {datetime.datetime.fromtimestamp(period1)} | Period2: {datetime.datetime.fromtimestamp(period2)}')
        query = f'{_YAHOO_API_URL}{code}.SA?period1={period1}&period2={period2}&interval=1d&events=history'
        if debug: print(f"Query: {query}")
        return query

    def get_yahoo_finance(tickers):
        dfs = []
        if not isinstance(tickers, list): tickers = [tickers]
        success = []
        errors = []
        for ticker in tickers:
            url = create_yahoo_download_query(code=ticker)
            try:
                tmp = pd.read_csv(url).assign(ticker=ticker)
                tmp = tmp.rename(columns={x: x.lower().replace(' ', '_') for x in tmp.columns}) # [_TICKERS_COLUMNS]
                tmp = tmp[tmp != 'null'].dropna()
                tmp = tmp.assign(
                    pct=(tmp['close'] - tmp['open']) / tmp['open'] * 100,
                    pct_ant=(tmp['close'] - tmp['close'].shift(1)) / tmp['close'].shift(1) * 100
                )
                # for col in _TICKERS_COLUMNS:
                #     tmp[col] = tmp[col].astype(float, errors='ignore')
                for col in tmp.columns:
                    tmp[col] = tmp[col].astype(float, errors='ignore')
                dfs.append(tmp)
                del(tmp)
                success.append(ticker)
            except Exception as exp:
                errors.append(ticker)
                print(f'{now()} [{ticker}] Error: {exp}')
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    dfs = []
    for ticker in tqdm(set(TICKERS)):
        tmp = get_yahoo_finance(tickers=ticker)
        if tmp.shape[0] > 1:
            tmp = tmp.rename(columns={'code': 'ticker'})
            tmp = create_ema(tmp.drop_duplicates(['date', 'ticker']))
            tmp = flag_volume(tmp)
            tmp['macd'] = macd(fast_ma=tmp['close_ema8'], slow_ma=tmp['close_ema20']).round(2)
            tmp['macd_signal'] = ema(serie=tmp['macd'], period=20)
            # tmp = get_signals(tmp)
            dfs.append(tmp)
            del(tmp)
    # Une os DataFrames
    df = pd.concat(dfs, ignore_index=True)

    # Cria colunas de interesse
    return df.assign(
        **{
            'candle_crossing_ema20': candle_crossing_ema(df, period=20),
            'crossing_8ema_x_20ema': np.where((df['close_ema8'] >= df['close_ema20']) & (df['close_ema8'].shift(1) < df['close_ema20'].shift(1)), 1, 0),
            'crossing_8ema_x_72ema': np.where((df['close_ema8'] >= df['close_ema72']) & (df['close_ema8'].shift(1) < df['close_ema72'].shift(1)), 1, 0),
            'crossing_20ema_x_72ema': np.where((df['close_ema20'] >= df['close_ema72']) & (df['close_ema20'].shift(1) < df['close_ema72'].shift(1)), 1, 0),
            'trend_tomorrow': np.where((df['close'].shift(-1).notna()) & (df['close'] < df['close'].shift(-1)), 1, 0),
            'volume_to_average': (df['volume'] / df['volume_ema20']),
            'macd_to_average': (df['macd'] / df['macd_signal']),
            'candle_lose': df['high'] - df['close'],
            'candle_gain': df['close'] - df['low'],
            'candle_length': df['high'] - df['low'],
            'price_var': df['close'] - df['open'],
            'candle_prop_close': 1 - ((df['high'] - df['low']) / df['close']),
            'price_prop_close': 1 - ((df['close'] - df['open']) / df['close']),
            'lower_shadow': (np.where(df['open'] < df['close'], df['open'], df['close']) - df['low']) / (df['high'] - df['low']),
            'upper_shadow': (df['high'] - np.where(df['open'] > df['close'], df['open'], df['close'])) / (df['high'] - df['low']),
            'ema8_over_20': df['close_ema8'] - df['close_ema20'],
            'ema8_over_72': df['close_ema8'] - df['close_ema72'],
            'ema20_over_72': df['close_ema20'] - df['close_ema72'],
            'prop_gain_8': (df['close'].shift(-8) - df['close']) * df['close'],
            'prop_gain_20': (df['close'].shift(-20) - df['close']) * df['close'],
            'prop_gain_72': (df['close'].shift(-72) - df['close']) * df['close'],
        }
    )