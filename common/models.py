# common/models.py

from pydantic import BaseModel, Field
from typing import List, Literal
from .config import (
    DEFAULT_ALPHABOT_LONG_SMA,
    DEFAULT_ALPHABOT_SHORT_SMA,
    DEFAULT_ALPHABOT_TRADE_QTY,
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
    DEFAULT_RISKGUARD_URL,
)

# --- Models for RiskGuard Agent ---


class TradeProposal(BaseModel):
    action: Literal["BUY", "SELL"]
    ticker: str
    quantity: int
    price: float


class PortfolioState(BaseModel):
    cash: float
    shares: int
    total_value: float


class RiskCheckPayload(BaseModel):
    trade_proposal: TradeProposal
    portfolio_state: PortfolioState
    max_pos_size: float = Field(default=DEFAULT_RISKGUARD_MAX_POS_SIZE)
    max_concentration: float = Field(default=DEFAULT_RISKGUARD_MAX_CONCENTRATION)


# --- Models for AlphaBot Agent ---


class AlphaBotPayload(BaseModel):
    historical_prices: List[float]
    current_price: float
    portfolio_state: PortfolioState
    short_sma_period: int = Field(default=DEFAULT_ALPHABOT_SHORT_SMA)
    long_sma_period: int = Field(default=DEFAULT_ALPHABOT_LONG_SMA)
    trade_quantity: int = Field(default=DEFAULT_ALPHABOT_TRADE_QTY)
    riskguard_url: str = Field(default=DEFAULT_RISKGUARD_URL)
    max_pos_size: float = Field(default=DEFAULT_RISKGUARD_MAX_POS_SIZE)
    max_concentration: float = Field(default=DEFAULT_RISKGUARD_MAX_CONCENTRATION)
    day: int
