"""
Centralized Configuration
All paths and settings in one place - Single database architecture
"""
from pathlib import Path

# Project root (auto-detected)
PROJECT_ROOT = Path(__file__).parent.parent

# Database directory
DB_DIR = PROJECT_ROOT / "db"
DB_DIR.mkdir(exist_ok=True)

# SINGLE DATABASE for everything
DATABASE_PATH = str(DB_DIR / "market_data.db")

# Aliases for code clarity (all point to same file)
MARKET_DATA_DB = DATABASE_PATH    # Historical prices + Greeks
OPTIONFLOW_DB = DATABASE_PATH     # Options flow (trades, OI, quotes, health)

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# IB Connection defaults
IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 238

# Collector defaults
COLLECTOR_DEFAULTS = {
    'symbols': 'SPX',
    'exchange_index': 'CBOE',
    'option_exchange': 'CBOE',
    'n_strikes': 30,
    'throttle_ms': 400,
    'spot_fallback': 6700.0,
    'oi_update_interval': 3600,
    'force_0dte': True
}

# Display
ENABLE_DEBUG = False

# Validate DB exists
if not Path(DATABASE_PATH).exists():
    print(f"⚠️  Database not found: {DATABASE_PATH}")