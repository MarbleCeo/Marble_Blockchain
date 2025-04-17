import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import random
from decimal import Decimal

# Import other modules as needed
# from src.dex import DEX
# from src.models import Order, Pool

def local_css(file_name):
    """Load and apply local CSS"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    # Apply CSS
    try:
        local_css("static/style.css")
    except Exception as e:
        st.error(f"Error loading CSS: {e}")
    
    # Set page config
    st.set_page_config(
        page_title="Marble DEX",
        page_icon="💎",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar for wallet connection and account info
    with st.sidebar:
        st.image("static/marble_logo.svg", width=200)
        st.title("Marble DEX")
        
        # Wallet connection
        if 'wallet_connected' not in st.session_state:
            st.session_state.wallet_connected = False
            
        if not st.session_state.wallet_connected:
            if st.button("Connect Wallet"):
                # This would actually connect to a wallet in production
                st.session_state.wallet_connected = True
                st.session_state.wallet_address = "0x" + "".join([random.choice("0123456789abcdef") for _ in range(40)])
                st.session_state.balance = {"ETH": 10.5, "USDT": 5000, "MARBLE": 1000}
                st.experimental_rerun()
        else:
            st.success(f"Connected: {st.session_state.wallet_address[:6]}...{st.session_state.wallet_address[-4:]}")
            st.subheader("Balances")
            for token, amount in st.session_state.balance.items():
                st.write(f"{token}: {amount}")
            
            if st.button("Disconnect"):
                st.session_state.wallet_connected = False
                if 'wallet_address' in st.session_state:
                    del st.session_state.wallet_address
                if 'balance' in st.session_state:
                    del st.session_state.balance
                st.experimental_rerun()
        
        # Market pairs selection
        st.subheader("Market Pairs")
        market_pairs = ["ETH/USDT", "MARBLE/USDT", "MARBLE/ETH"]
        selected_pair = st.selectbox("Select Trading Pair", market_pairs, index=1)
        st.session_state.base_token, st.session_state.quote_token = selected_pair.split("/")

    # Main content
    tab1, tab2, tab3 = st.tabs(["Trade", "Liquidity Pools", "Your Orders"])
    
    # Trade tab
    with tab1:
        # Top row with price information
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric(
                label=f"Current {selected_pair} Price", 
                value=f"${generate_mock_price(selected_pair)}", 
                delta=f"{random.choice(['+', '-'])}{random.uniform(0.1, 2.5):.2f}%"
            )
        
        with col2:
            st.metric(
                label="24h Volume", 
                value=f"${random.uniform(100000, 5000000):.2f}", 
            )
            
        with col3:
            st.metric(
                label="24h Change", 
                value=f"{random.choice(['+', '-'])}{random.uniform(0.1, 8.5):.2f}%", 
            )
        
        # Trading interface
        col_chart, col_book = st.columns([3, 1])
        
        # Price chart
        with col_chart:
            st.subheader("Price Chart")
            chart_container = st.container()
            with chart_container:
                fig = create_candlestick_chart(selected_pair)
                st.plotly_chart(fig, use_container_width=True)
        
        # Order book
        with col_book:
            st.subheader("Order Book")
            
            # Sell orders (asks) - display in reverse order (highest first)
            asks = generate_mock_orders(selected_pair, "sell", 8)
            asks_df = pd.DataFrame(asks)
            asks_df = asks_df.sort_values(by="price", ascending=False)
            
            st.markdown('<div class="order-book asks">', unsafe_allow_html=True)
            for _, row in asks_df.iterrows():
                price_color = "red-text"
                st.markdown(
                    f'<div class="order-row">'
                    f'<span class="{price_color}">${row["price"]:.2f}</span>'
                    f'<span class="amount">{row["amount"]:.4f}</span>'
                    f'<span class="total">${row["total"]:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Current price indicator
            current_price = generate_mock_price(selected_pair)
            st.markdown(
                f'<div class="current-price">'
                f'<span>${current_price}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Buy orders (bids)
            bids = generate_mock_orders(selected_pair, "buy", 8)
            bids_df = pd.DataFrame(bids)
            bids_df = bids_df.sort_values(by="price", ascending=False)
            
            st.markdown('<div class="order-book bids">', unsafe_allow_html=True)
            for _, row in bids_df.iterrows():
                price_color = "green-text"
                st.markdown(
                    f'<div class="order-row">'
                    f'<span class="{price_color}">${row["price"]:.2f}</span>'
                    f'<span class="amount">{row["amount"]:.4f}</span>'
                    f'<span class="total">${row["total"]:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Buy/Sell interface
        st.subheader("Place Order")
        col_buy, col_sell = st.columns(2)
        
        with col_buy:
            st.markdown('<div class="order-form buy-form">', unsafe_allow_html=True)
            st.subheader("Buy")
            buy_price = st.number_input(f"Price ({st.session_state.quote_token})", min_value=0.0, value=float(current_price), key="buy_price")
            buy_amount = st.number_input(f"Amount ({st.session_state.base_token})", min_value=0.0, value=1.0, key="buy_amount")
            buy_total = buy_price * buy_amount
            st.write(f"Total: {buy_total:.2f} {st.session_state.quote_token}")
            
            buy_options = st.selectbox("Order Type", ["Market", "Limit"], key="buy_order_type")
            
            if st.button("Buy", key="buy_button", type="primary"):
                if st.session_state.wallet_connected:
                    st.success(f"Order placed: BUY {buy_amount} {st.session_state.base_token} at {buy_price} {st.session_state.quote_token}")
                else:
                    st.error("Please connect your wallet first")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_sell:
            st.markdown('<div class="order-form sell-form">', unsafe_allow_html=True)
            st.subheader("Sell")
            sell_price = st.number_input(f"Price ({st.session_state.quote_token})", min_value=0.0, value=float(current_price), key="sell_price")
            sell_amount = st.number_input(f"Amount ({st.session_state.base_token})", min_value=0.0, value=1.0, key="sell_amount")
            sell_total = sell_price * sell_amount
            st.write(f"Total: {sell_total:.2f} {st.session_state.quote_token}")
            
            sell_options = st.selectbox("Order Type", ["Market", "Limit"], key="sell_order_type")
            
            if st.button("Sell", key="sell_button", type="primary"):
                if st.session_state.wallet_connected:
                    st.success(f"Order placed: SELL {sell_amount} {st.session_state.base_token} at {sell_price} {st.session_state.quote_token}")
                else:
                    st.error("Please connect your wallet first")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Liquidity Pools tab
    with tab2:
        st.subheader("Liquidity Pools")
        
        # Display pool information
        pools_data = [
            {"pair": "ETH/USDT", "liquidity": "$4,532,000", "volume_24h": "$1,250,000", "apy": "12.5%"},
            {"pair": "MARBLE/USDT", "liquidity": "$1,250,000", "volume_24h": "$450,000", "apy": "18.2%"},
            {"pair": "MARBLE/ETH", "liquidity": "$850,000", "volume_24h": "$320,000", "apy": "15.7%"},
        ]
        
        pools_df = pd.DataFrame(pools_data)
        st.dataframe(pools_df, use_container_width=True)
        
        # Add liquidity interface
        st.subheader("Add Liquidity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            liq_pair = st.selectbox("Select Pair", ["ETH/USDT", "MARBLE/USDT", "MARBLE/ETH"])
            token1, token2 = liq_pair.split('/')
            
            token1_amount = st.number_input(f"{token1} Amount", min_value=0.0, value=1.0)
            token2_amount = st.number_input(f"{token2} Amount", min_value=0.0, value=calculate_token2_amount(token1, token2, token1_amount))
            
        with col2:
            st.write("Liquidity Pool Share")
            current_pool = next((p for p in pools_data if p["pair"] == liq_pair), None)
            if current_pool:
                # Convert string to float without $ and comma
                liquidity_value = float(current_pool["liquidity"].replace("$", "").replace(",", ""))
                contribution = (token1_amount * get_token_price(token1)) * 2
                share_percentage = (contribution / liquidity_value) * 100
                st.metric("Your Share", f"{share_percentage:.2f}%")
                st.metric("Pool APY", current_pool["apy"])
            
            if st.button("Add Liquidity", type="primary"):
                if st.session_state.wallet_connected:
                    st.success(f"Added liquidity: {token1_amount} {token1} and {token2_amount} {token2}")
                else:
                    st.error("Please connect your wallet first")
    
    # Your Orders tab
    with tab3:
        st.subheader("Your Orders")
        
        if st.session_state.wallet_connected:
            # Generate mock orders for the user
            user_orders = [
                {"id": "0x1234...", "pair": "MARBLE/USDT", "type": "BUY", "price": "$1.25", "amount": "100 MARBLE", "total": "$125", "status": "Open", "time": "2 mins ago"},
                {"id": "0x2345...", "pair": "ETH/USDT", "type": "SELL", "price": "$2800", "amount": "0.5 ETH", "total": "$1400", "status": "Filled", "time": "1 hour ago"},
                {"id": "0x3456...", "pair": "MARBLE/ETH", "type": "BUY", "price": "0.0004 ETH", "amount": "500 MARBLE", "total": "0.2 ETH", "status": "Partial", "time": "3 hours ago"},
            ]
            
            orders_df = pd.DataFrame(user_orders)
            st.dataframe(orders_df, use_container_width=True)
            
            # Cancel order button
            cancel_order_id = st.selectbox("Select Order to Cancel", [order["id"] for order in user_orders if order["status"] in ["Open", "Partial"]])
            if st.button("Cancel Order"):
                st.success(f"Order {cancel_order_id} cancelled successfully")
        else:
            st.info("Connect your wallet to view your orders")


# Helper functions for generating mock data
def generate_mock_price(pair):
    """Generate a realistic price for the given pair"""
    base_prices = {
        "ETH/USDT": 2800.0,
        "MARBLE/USDT": 1.25,
        "MARBLE/ETH": 0.00045,
    }
    base_price = base_prices.get(pair, 1.0)
    variation = random.uniform(-0.02, 0.02)  # 2% variation
    return round(base_price * (1 + variation), 4)

def generate_mock_orders(pair, order_type, count=10):
    """Generate mock orders for the order book"""
    base_price = generate_mock_price(pair)
    orders = []
    
    for i in range(count):
        if order_type == "buy":
            # Buyers want to buy at a lower price
            price_offset = random.uniform(0.01, 0.1) * (i + 1)
            price = base_price * (1 - price_offset)
        else:
            # Sellers want to sell at a higher price
            price_offset = random.uniform(0.01, 0.1) * (i + 1)
            price = base_price * (1 + price_offset)
        
        amount = random.uniform(0.1, 10)
        total = price * amount
        
        orders.append({
            "price": price,
            "amount": amount,
            "total": total
        })
    
    return orders

def create_candlestick_chart(pair):
    """Create a mock candlestick chart for the selected pair"""
    base_price = generate_mock_price(pair)
    
    # Generate date range
    end_date = pd.Timestamp.now()
    date_range = pd.date

import streamlit as st
import pandas as pd
import requests
import json
import time
import plotly.graph_objects as go
from datetime import datetime

# Constants
API_URL = "http://localhost:8000"
LOGO_PATH = "../static/marble_logo.svg"

# Initialize session state
if 'connected_wallet' not in st.session_state:
    st.session_state.connected_wallet = None
if 'selected_pair' not in st.session_state:
    st.session_state.selected_pair = "ETH-USDT"
if 'buy_price' not in st.session_state:
    st.session_state.buy_price = 0.0
if 'buy_amount' not in st.session_state:
    st.session_state.buy_amount = 0.0
if 'sell_price' not in st.session_state:
    st.session_state.sell_price = 0.0
if 'sell_amount' not in st.session_state:
    st.session_state.sell_amount = 0.0
if 'order_type' not in st.session_state:
    st.session_state.order_type = "Market"

# Helper functions
def format_number(num, decimals=2):
    """Format number with commas and specified decimals"""
    return f"{num:,.{decimals}f}"

def get_orderbook(pair):
    """Fetch orderbook from API"""
    try:
        response = requests.get(f"{API_URL}/api/v1/orderbook/{pair}")
        return response.json()
    except:
        st.error("Failed to fetch orderbook data")
        return {"bids": [], "asks": []}

def get_trading_history(pair):
    """Fetch trading history from API"""
    try:
        response = requests.get(f"{API_URL}/api/v1/history/{pair}")
        return response.json()["history"]
    except:
        st.error("Failed to fetch trading history")
        return []

def get_liquidity_pools():
    """Fetch liquidity pools from API"""
    try:
        response = requests.get(f"{API_URL}/api/v1/pools")
        return response.json()["pools"]
    except:
        st.error("Failed to fetch liquidity pools")
        return {}

def connect_wallet(address, signature):
    """Connect wallet to the DEX"""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/wallet/connect",
            json={"address": address, "signature": signature}
        )
        return response.json()["wallet"]
    except:
        st.error("Failed to connect wallet")
        return None

def place_order(user_address, pair, order_type, amount, price):
    """Place a new order"""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/orders",
            json={
                "user_address": user_address,
                "pair": pair,
                "type": order_type.lower(),
                "amount": amount,
                "price": price
            }
        )
        return response.json()
    except:
        st.error("Failed to place order")
        return None

def add_liquidity(user_address, pair, token1_amount, token2_amount):
    """Add liquidity to a pool"""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/pools/add-liquidity",
            json={
                "user_address": user_address,
                "pair": pair,
                "token1_amount": token1_amount,
                "token2_amount": token2_amount
            }
        )
        return response.json()
    except:
        st.error("Failed to add liquidity")
        return None

# UI Functions
def render_header():
    """Render the application header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.image("static/marble_logo.svg", width=80)
    
    with col2:
        st.title("Marble DEX")
        st.markdown("#### Decentralized Exchange Platform")
    
    with col3:
        if st.session_state.connected_wallet:
            st.success(f"Connected: {st.session_state.connected_wallet['address'][:6]}...{st.session_state.connected_wallet['address'][-4:]}")
            if st.button("Disconnect"):
                st.session_state.connected_wallet = None
                st.experimental_rerun()
        else:
            # In a real app, this would integrate with web3 wallets
            example_addresses = [
                "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                "0x1234567890123456789012345678901234567890",
                "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
            ]
            wallet_address = st.selectbox("Select a demo wallet", example_addresses)
            if st.button("Connect

