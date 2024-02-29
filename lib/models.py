URL_LIB = 'https://raw.githubusercontent.com/Genicleito/market_trading_analysis/master/lib/technical_analysis/__init__.py'

# Paths
LIB_PATH = 'lib/technical_analysis.py'
READ_MARKET_DATA_PATH = 'data/raw/hist_market_trading_yfinance.csv.zip'

# Constantes
P = 0.05
PERIODS_H_TEST = 21
PERIODS_ANT_PREDICTION = 200
PERIODS_FORECAST = 20

def get_widget_trading_view(ticker, interval='1D'):
    return """<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <div class="tradingview-widget-copyright"><a href="https://br.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Monitore todos os mercados no TradingView</span></a></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js" async>
  {
  "symbols": [
    [
      "BMFBOVESPA:_ticker_|7D"
    ]
  ],
  "chartOnly": false,
  "width": 550,
  "height": 400,
  "locale": "br",
  "colorTheme": "dark",
  "autosize": false,
  "showVolume": true,
  "showMA": true,
  "hideDateRanges": false,
  "hideMarketStatus": false,
  "hideSymbolLogo": false,
  "scalePosition": "right",
  "scaleMode": "Normal",
  "fontFamily": "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
  "fontSize": "10",
  "noTimeScale": false,
  "valuesTracking": "1",
  "changeMode": "price-and-percent",
  "chartType": "candlesticks",
  "maLineColor": "rgba(186, 104, 200, 1)",
  "maLineWidth": 1,
  "maLength": 8,
  "lineType": 0,
  "dateRanges": [
    "1d|15",
    "1w|1D",
    "1m|1D",
    "3m|1D",
    "12m|1D",
    "60m|1M",
    "all|1M"
  ],
  "upColor": "#22ab94",
  "downColor": "#f7525f",
  "borderUpColor": "#22ab94",
  "borderDownColor": "#f7525f",
  "wickUpColor": "#22ab94",
  "wickDownColor": "#f7525f"
}
  </script>
</div>
<!-- TradingView Widget END -->""".replace('_ticker_', ticker)