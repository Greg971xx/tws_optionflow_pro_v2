from ib_insync import Stock, Index, Future

# Liste centralisée des actifs proposés à l'utilisateur
# Ils seront testés plus tard avec reqContractDetails ou reqHistoricalData

TICKERS_CATALOG = {
    # Indices US & Vol
    "SPX": Index("SPX", "CBOE", currency="USD"),
    "NDX": Index("NDX", "NASDAQ", currency="USD"),
    "RUT": Index("RUT", "RUSSELL", currency="USD"),
    "VIX": Index("VIX", "CBOE", currency="USD"),
    "VXN": Index("VXN", "CBOE", currency="USD"),
    "RVX": Index("RVX", "CBOE", currency="USD"),
    #"YM": Index("RVX", "CBOT", currency="USD"),

    # Futures indices
    "ES": Future(symbol="ES", exchange="CME", currency="USD", lastTradeDateOrContractMonth="202509"),
    "NQ": Future(symbol="NQ", exchange="CME", currency="USD", lastTradeDateOrContractMonth="202509"),
    "RTY": Future(symbol="RTY", exchange="CME", currency="USD", lastTradeDateOrContractMonth="202509"),
    "YM": Future(symbol="YM", exchange="CBOT", currency="USD", lastTradeDateOrContractMonth="202509"),



    # Futures matières premières
    "GC": Future(symbol="GC", exchange="COMEX", currency="USD", lastTradeDateOrContractMonth="202509"),
    "CL": Future(symbol="CL", exchange="NYMEX", currency="USD", lastTradeDateOrContractMonth="202509"),
    "SI": Future("SI", exchange="COMEX", currency="USD", lastTradeDateOrContractMonth="202509", tradingClass="SI", multiplier="5000"),




    # ETFs US
    "SPY": Stock("SPY", "SMART", currency="USD"),
    "QQQ": Stock("QQQ", "SMART", currency="USD"),
    "IWM": Stock("IWM", "SMART", currency="USD"),
    "GLD": Stock("GLD", "SMART", currency="USD"),
    "SLV": Stock("SLV", "SMART", currency="USD"),
    "USO": Stock("USO", "SMART", currency="USD"),
    "UUP": Stock("UUP", "SMART", currency="USD"),
    "FXE": Stock("FXE", "SMART", currency="USD"),
    "FXY": Stock("FXY", "SMART", currency="USD"),

    # Tech US (MAC7)
    "AAPL": Stock("AAPL", "SMART", currency="USD"),
    "AMZN": Stock("AMZN", "SMART", currency="USD"),
    "META": Stock("META", "SMART", currency="USD"),
    "GOOGL": Stock("GOOGL", "SMART", currency="USD"),
    "MSFT": Stock("MSFT", "SMART", currency="USD"),
    "NVDA": Stock("NVDA", "SMART", currency="USD"),
    "TSLA": Stock("TSLA", "SMART", currency="USD"),

    # Europe
    "ESTX50": Index("ESTX50", "EUREX", currency="EUR"),
    "DAX": Index("DAX", "EUREX", currency="EUR"),
    "CAC40": Index("CAC40", "MONEP", currency="EUR"),
    "V2TX": Index("V2TX", "EUREX", currency="EUR"),
}
