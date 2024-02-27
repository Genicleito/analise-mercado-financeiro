import pandas as pd
import numpy as np
import scipy
import datetime

def now(as_date_text=False):
    if as_date_text:
        return datetime.datetime.now().strftime('%Y-%m-%d')
    return datetime.datetime.now()

def cox_stuart_test(X, p=0.05, trend_type='i', debug=True):
    """Função pararealizar o teste de hipótese de Cox-Stuart.

    Parameters:
        * `X` list: Vetor com os valores numéricos.
        * `p` float: Valor de probabilidade p para teste de hipótese. Default: `0.05`.
        * `trend_type` str: Tipo da tendência a ser calculada, sendo `'d'` *decreasing trend* e
            `'i'` *increasing trend*.
    """
    ts = now()
    # Divide os dados (analisa se o conjunto de dados tem tamanho par ou ímpar)
    if len(X) % 2 == 0:
        # threshold = (X[(len(X) // 2) - 1] + X[len(X) // 2]) / 2
        x1 = X[ :len(X) // 2 ]
        x2 = X[(len(X) // 2): ]
    else:
        # threshold = X[len(X) // 2]
        x1 = X[ :len(X) // 2]
        x2 = X[(len(X) // 2) + 1: ]

    # Difference
    difference = x1 - x2

    # Signal
    signs = np.sign(difference)
    # # Signs not equal to 0
    # signs = signs[signs != 0]

    # Increase and decrease values
    pos = signs[signs > 0]
    neg = signs[signs < 0]

    # Length
    n = len(pos) + len(neg)

    if trend_type == 'd':
        x = len(neg)
        p_value = scipy.stats.binom.cdf(x, n, p)
        if debug: print(f"x = {x}\nn = {n}\np = {p}")
        if debug: print(f"P-Value: {p_value}\n\t{'Não há tendência sgnificativa de baixa!' if p_value >= p else 'Há tendência significativa de baixa!'}")
    else: # 'i'
        x = len(pos)
        p_value = scipy.stats.binom.cdf(x, n, p)
        if debug: print(f"x = {x}\nn = {n}\np = {p}")
        if debug: print(f"P-Value: {p_value}\n\t{'Não há tendência sgnificativa de alta!' if p_value >= p else 'Há tendência significativa de alta!'}")
    
    if debug: print(f"Tempo de execução: {now() - ts}")
    return p_value, p_value < p

