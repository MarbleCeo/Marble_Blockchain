from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum as SQLEnum, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from src.config import DATABASE_URL

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    wallet_address = Column(String, unique=True, index=True)
    username = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    balances = relationship("Balance", back_populates="user")
    orders = relationship("Order", back_populates="user")


class Token(Base):
    __tablename__ = "tokens"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    symbol = Column(String, unique=True, index=True)
    decimals = Column(Integer, default=18)
    address = Column(String, unique=True, index=True)
    
    balances = relationship("Balance", back_populates="token")
    base_pairs = relationship("TradingPair", foreign_keys="TradingPair.base_token_id", back_populates="base_token")
    quote_pairs = relationship("TradingPair", foreign_keys="TradingPair.quote_token_id", back_populates="quote_token")


class Balance(Base):
    __tablename__ = "balances"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token_id = Column(Integer, ForeignKey("tokens.id"))
    amount = Column(Float, default=0.0)
    locked_amount = Column(Float, default=0.0)
    
    user = relationship("User", back_populates="balances")
    token = relationship("Token", back_populates="balances")


class TradingPair(Base):
    __tablename__ = "trading_pairs"

    id = Column(Integer, primary_key=True, index=True)
    base_token_id = Column(Integer, ForeignKey("tokens.id"))
    quote_token_id = Column(Integer, ForeignKey("tokens.id"))
    is_active = Column(Boolean, default=True)
    
    base_token = relationship("Token", foreign_keys=[base_token_id], back_populates="base_pairs")
    quote_token = relationship("Token", foreign_keys=[quote_token_id], back_populates="quote_pairs")
    orders = relationship("Order", back_populates="trading_pair")
    liquidity_pool = relationship("LiquidityPool", back_populates="trading_pair", uselist=False)


class LiquidityPool(Base):
    __tablename__ = "liquidity_pools"

    id = Column(Integer, primary_key=True, index=True)
    trading_pair_id = Column(Integer, ForeignKey("trading_pairs.id"), unique=True)
    base_token_reserve = Column(Float, default=0.0)
    quote_token_reserve = Column(Float, default=0.0)
    total_lp_tokens = Column(Float, default=0.0)
    
    trading_pair = relationship("TradingPair", back_populates="liquidity_pool")
    lp_positions = relationship("LiquidityPosition", back_populates="liquidity_pool")


class LiquidityPosition(Base):
    __tablename__ = "liquidity_positions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    liquidity_pool_id = Column(Integer, ForeignKey("liquidity_pools.id"))
    lp_tokens = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User")
    liquidity_pool = relationship("LiquidityPool", back_populates="lp_positions")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    trading_pair_id = Column(Integer, ForeignKey("trading_pairs.id"))
    type = Column(SQLEnum(OrderType))
    side = Column(SQLEnum(OrderSide))
    amount = Column(Float)
    filled_amount = Column(Float, default=0.0)
    price = Column(Float, nullable=True)  # Null for market orders
    status = Column(SQLEnum(OrderStatus), default=OrderStatus.OPEN)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="orders")
    trading_pair = relationship("TradingPair", back_populates="orders")
    trades = relationship("Trade", back_populates="order")


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    maker_order_id = Column(Integer, ForeignKey("orders.id"), nullable=True)
    amount = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    order = relationship("Order", foreign_keys=[order_id], back_populates="trades")
    maker_order = relationship("Order", foreign_keys=[maker_order_id])


# Create all tables in the database
def create_tables():
    Base.metadata.create_all(bind=engine)

