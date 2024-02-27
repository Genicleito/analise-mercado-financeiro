import pandas as pd
import numpy as np
import scipy
import datetime

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
