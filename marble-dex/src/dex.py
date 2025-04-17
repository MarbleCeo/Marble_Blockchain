from decimal import Decimal
import math
from typing import List, Tuple, Dict, Optional
from sqlalchemy.orm import Session

from src.models import (
    Token, TradingPair, Order, OrderType, OrderSide, OrderStatus,
    LiquidityPool, LiquidityPosition, Trade, User, Balance
)
from src.config import LIQUIDITY_PROVIDER_FEE, PROTOCOL_FEE


class DEXEngine:
    def __init__(self, db: Session):
        self.db = db
        self._order_books = {}  # Cache for order books

    def get_token_by_symbol(self, symbol: str) -> Optional[Token]:
        """Get a token by its symbol."""
        return self.db.query(Token).filter(Token.symbol == symbol).first()

    def get_trading_pair(self, base_symbol: str, quote_symbol: str) -> Optional[TradingPair]:
        """Get a trading pair by base and quote token symbols."""
        base_token = self.get_token_by_symbol(base_symbol)
        quote_token = self.get_token_by_symbol(quote_symbol)
        
        if not base_token or not quote_token:
            return None
            
        return self.db.query(TradingPair).filter(
            TradingPair.base_token_id == base_token.id,
            TradingPair.quote_token_id == quote_token.id
        ).first()

    def get_user_balance(self, wallet_address: str, token_symbol: str) -> Optional[Balance]:
        """Get a user's balance for a specific token."""
        user = self.db.query(User).filter(User.wallet_address == wallet_address).first()
        token = self.get_token_by_symbol(token_symbol)
        
        if not user or not token:
            return None
            
        return self.db.query(Balance).filter(
            Balance.user_id == user.id,
            Balance.token_id == token.id
        ).first()

    def place_order(self, 
                   wallet_address: str, 
                   base_symbol: str, 
                   quote_symbol: str, 
                   order_type: OrderType, 
                   side: OrderSide, 
                   amount: float, 
                   price: Optional[float] = None) -> Optional[Order]:
        """Place a new order."""
        # Get the trading pair
        trading_pair = self.get_trading_pair(base_symbol, quote_symbol)
        if not trading_pair:
            return None
            
        # Get or create user
        user = self.db.query(User).filter(User.wallet_address == wallet_address).first()
        if not user:
            user = User(wallet_address=wallet_address)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            
        # Validate order parameters
        if order_type == OrderType.LIMIT and not price:
            return None
            
        if amount <= 0:
            return None
            
        # Check balance
        token_to_check = base_symbol if side == OrderSide.SELL else quote_symbol
        required_amount = amount if side == OrderSide.SELL else (amount * price if price else 0)
        
        balance = self.get_user_balance(wallet_address, token_to_check)
        if not balance or balance.amount < required_amount:
            return None
            
        # Lock funds
        balance.locked_amount += required_amount
        balance.amount -= required_amount
        
        # Create the order
        new_order = Order(
            user_id=user.id,
            trading_pair_id=trading_pair.id,
            type=order_type,
            side=side,
            amount=amount,
            price=price,
            status=OrderStatus.OPEN
        )
        
        self.db.add(new_order)
        self.db.commit()
        self.db.refresh(new_order)
        
        # Process the order (match with existing orders or add to order book)
        if order_type == OrderType.MARKET or price:  # Market orders or limit orders with price
            self._process_order(new_order)
            
        return new_order

    def _process_order(self, order: Order) -> None:
        """Process an order by matching it with existing orders or adding it to the order book."""
        if order.type == OrderType.MARKET:
            self._match_market_order(order)
        else:  # Limit order
            self._match_limit_order(order)
            
        # Update order status
        if order.filled_amount >= order.amount:
            order.status = OrderStatus.FILLED
        elif order.filled_amount > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
            
        self.db.commit()

    def _match_market_order(self, order: Order) -> None:
        """Match a market order with existing limit orders."""
        trading_pair = self.db.query(TradingPair).get(order.trading_pair_id)
        opposite_side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
        
        # Get opposite orders sorted by price (best price first)
        opposite_orders = self.db.query(Order).filter(
            Order.trading_pair_id == order.trading_pair_id,
            Order.side == opposite_side,
            Order.status.in_([OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]),
            Order.type == OrderType.LIMIT
        )
        
        if order.side == OrderSide.BUY:
            opposite_orders = opposite_orders.order_by(Order.price.asc())
        else:
            opposite_orders = opposite_orders.order_by(Order.price.desc())
            
        

