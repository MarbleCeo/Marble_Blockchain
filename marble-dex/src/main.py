from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import json
from datetime import datetime
import uuid

app = FastAPI(title="Marble DEX API", description="Decentralized Exchange API for the Marble DEX platform")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock database - in a real implementation, use a database
class MockDB:
    def __init__(self):
        self.orders = []
        self.liquidity_pools = {
            "ETH-USDT": {
                "token1_amount": 100.0,
                "token2_amount": 200000.0,
                "token1_symbol": "ETH",
                "token2_symbol": "USDT",
                "pool_address": "0x123456789abcdef",
                "apy": 5.2
            },
            "BTC-USDT": {
                "token1_amount": 10.0,
                "token2_amount": 300000.0,
                "token1_symbol": "BTC", 
                "token2_symbol": "USDT",
                "pool_address": "0xabcdef123456789",
                "apy": 4.8
            },
            "SOL-USDT": {
                "token1_amount": 1000.0,
                "token2_amount": 100000.0,
                "token1_symbol": "SOL",
                "token2_symbol": "USDT",
                "pool_address": "0x987654321abcdef",
                "apy": 6.5
            }
        }
        self.wallets = {}

db = MockDB()

# Models
class Order(BaseModel):
    id: Optional[str] = None
    user_address: str
    pair: str
    type: str  # "buy" or "sell"
    amount: float
    price: float
    status: str = "open"
    timestamp: Optional[datetime] = None

class LiquidityProvision(BaseModel):
    user_address: str
    pair: str
    token1_amount: float
    token2_amount: float

class WalletConnection(BaseModel):
    address: str
    signature: str

class Token(BaseModel):
    symbol: str
    name: str
    balance: float
    price: float

class Wallet(BaseModel):
    address: str
    tokens: List[Token]
    connected: bool = False

# Routes
@app.get("/")
async def read_root():
    return {"message": "Welcome to Marble DEX API"}

# Order Book Endpoints
@app.get("/api/v1/orderbook/{pair}")
async def get_orderbook(pair: str):
    if pair not in ["ETH-USDT", "BTC-USDT", "SOL-USDT"]:
        raise HTTPException(status_code=404, detail="Trading pair not found")
    
    # Filter orders for the requested pair
    buy_orders = [order for order in db.orders if order.pair == pair and order.type == "buy" and order.status == "open"]
    sell_orders = [order for order in db.orders if order.pair == pair and order.type == "sell" and order.status == "open"]
    
    # Sort orders appropriately
    buy_orders.sort(key=lambda x: x.price, reverse=True)  # Highest buy price first
    sell_orders.sort(key=lambda x: x.price)  # Lowest sell price first
    
    return {
        "pair": pair,
        "bids": [{"price": order.price, "amount": order.amount, "total": order.price * order.amount} for order in buy_orders],
        "asks": [{"price": order.price, "amount": order.amount, "total": order.price * order.amount} for order in sell_orders]
    }

@app.post("/api/v1/orders")
async def create_order(order: Order):
    # Generate order ID and timestamp
    order.id = str(uuid.uuid4())
    order.timestamp = datetime.now()
    
    # In a real implementation, validate the order, check balances, etc.
    
    # Add to the order book
    db.orders.append(order)
    
    # In a real implementation, match orders and execute trades
    
    return {"order_id": order.id, "status": "open"}

@app.get("/api/v1/orders/{user_address}")
async def get_user_orders(user_address: str):
    user_orders = [order for order in db.orders if order.user_address == user_address]
    return {"orders": user_orders}

@app.delete("/api/v1/orders/{order_id}")
async def cancel_order(order_id: str):
    for i, order in enumerate(db.orders):
        if order.id == order_id:
            db.orders[i].status = "cancelled"
            return {"message": f"Order {order_id} cancelled successfully"}
    
    raise HTTPException(status_code=404, detail="Order not found")

# Liquidity Pool Endpoints
@app.get("/api/v1/pools")
async def get_liquidity_pools():
    return {"pools": db.liquidity_pools}

@app.get("/api/v1/pools/{pair}")
async def get_liquidity_pool(pair: str):
    if pair not in db.liquidity_pools:
        raise HTTPException(status_code=404, detail="Liquidity pool not found")
    
    return {"pool": db.liquidity_pools[pair]}

@app.post("/api/v1/pools/add-liquidity")
async def add_liquidity(liquidity: LiquidityProvision):
    if liquidity.pair not in db.liquidity_pools:
        raise HTTPException(status_code=404, detail="Liquidity pool not found")
    
    # In a real implementation, validate the amounts, check balances, etc.
    
    # Update pool amounts
    pool = db.liquidity_pools[liquidity.pair]
    pool["token1_amount"] += liquidity.token1_amount
    pool["token2_amount"] += liquidity.token2_amount
    
    return {"message": "Liquidity added successfully", "updated_pool": pool}

@app.post("/api/v1/pools/remove-liquidity")
async def remove_liquidity(liquidity: LiquidityProvision):
    if liquidity.pair not in db.liquidity_pools:
        raise HTTPException(status_code=404, detail="Liquidity pool not found")
    
    # In a real implementation, validate the amounts, check balances, etc.
    
    # Update pool amounts
    pool = db.liquidity_pools[liquidity.pair]
    
    if liquidity.token1_amount > pool["token1_amount"] or liquidity.token2_amount > pool["token2_amount"]:
        raise HTTPException(status_code=400, detail="Insufficient liquidity in pool")
    
    pool["token1_amount"] -= liquidity.token1_amount
    pool["token2_amount"] -= liquidity.token2_amount
    
    return {"message": "Liquidity removed successfully", "updated_pool": pool}

# Wallet Endpoints
@app.post("/api/v1/wallet/connect")
async def connect_wallet(wallet_connection: WalletConnection):
    # In a real implementation, verify the signature
    
    # Create or update wallet
    if wallet_connection.address not in db.wallets:
        db.wallets[wallet_connection.address] = Wallet(
            address=wallet_connection.address,
            tokens=[
                Token(symbol="ETH", name="Ethereum", balance=10.0, price=2000.0),
                Token(symbol="BTC", name="Bitcoin", balance=0.5, price=30000.0),
                Token(symbol="SOL", name="Solana", balance=100.0, price=100.0),
                Token(symbol="USDT", name="Tether", balance=20000.0, price=1.0)
            ],
            connected=True
        )
    else:
        db.wallets[wallet_connection.address].connected = True
    
    return {"message": "Wallet connected successfully", "wallet": db.wallets[wallet_connection.address]}

@app.post("/api/v1/wallet/disconnect")
async def disconnect_wallet(wallet_connection: WalletConnection):
    if wallet_connection.address in db.wallets:
        db.wallets[wallet_connection.address].connected = False
        return {"message": "Wallet disconnected successfully"}
    
    raise HTTPException(status_code=404, detail="Wallet not found")

@app.get("/api/v1/wallet/{address}")
async def get_wallet(address: str):
    if address in db.wallets:
        return {"wallet": db.wallets[address]}
    
    raise HTTPException(status_code=404, detail="Wallet not found")

# Trading History Endpoints
@app.get("/api/v1/history/{pair}")
async def get_trading_history(pair: str):
    # Mock trading history
    history = [
        {"time": "2023-05-01T12:00:00", "price": 1950.0, "amount": 0.5, "type": "buy"},
        {"time": "2023-05-01T12:15:00", "price": 1955.0, "amount": 0.3, "type": "sell"},
        {"time": "2023-05-01T12:30:00", "price": 1960.0, "amount": 0.7, "type": "buy"},
        {"time": "2023-05-01T12:45:00", "price": 1958.0, "amount": 0.4, "type": "sell"},
        {"time": "2023-05-01T13:00:00", "price": 1965.0, "amount": 0.6, "type": "buy"}
    ]
    
    return {"pair": pair, "history": history}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

