import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dex.db")

# Web3 configuration
WEB3_PROVIDER_URL = os.getenv("WEB3_PROVIDER_URL", "http://localhost:8545")

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# UI configuration
UI_PORT = int(os.getenv("UI_PORT", "8501"))

# Token configuration
SUPPORTED_TOKENS = {
    "ETH": {
        "name": "Ethereum",
        "symbol": "ETH",
        "decimals": 18,
        "address": "0x0000000000000000000000000000000000000000"  # Native token
    },
    "USDT": {
        "name": "Tether USD",
        "symbol": "USDT",
        "decimals": 6,
        "address": os.getenv("USDT_ADDRESS", "0xdAC17F958D2ee523a2206206994597C13D831ec7")
    },
    "USDC": {
        "name": "USD Coin",
        "symbol": "USDC",
        "decimals": 6,
        "address": os.getenv("USDC_ADDRESS", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
    },
    "DAI": {
        "name": "Dai Stablecoin",
        "symbol": "DAI",
        "decimals": 18,
        "address": os.getenv("DAI_ADDRESS", "0x6B175474E89094C44Da98b954EedeAC495271d0F")
    },
    "MBL": {
        "name": "Marble Token",
        "symbol": "MBL",
        "decimals": 18,
        "address": os.getenv("MBL_ADDRESS", "0x0000000000000000000000000000000000000001")  # Placeholder
    }
}

# Default trading pairs
DEFAULT_PAIRS = [
    ("ETH", "USDT"),
    ("ETH", "USDC"),
    ("ETH", "DAI"),
    ("MBL", "ETH"),
    ("MBL", "USDT")
]

# Fee configuration
LIQUIDITY_PROVIDER_FEE = 0.003  # 0.3%
PROTOCOL_FEE = 0.001  # 0.1%

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "marble_dex_secret_key")

