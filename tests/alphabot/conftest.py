import pytest
from typing import Callable
from alphabot.agent import AlphaBotAgent
from common.models import AlphaBotPayload, PortfolioState


@pytest.fixture
def agent() -> AlphaBotAgent:
    """Provides an AlphaBotAgent instance."""
    return AlphaBotAgent(stock_ticker="TEST")


@pytest.fixture
def historical_prices_buy_signal() -> list[float]:
    """Provides a list of historical prices that will trigger a BUY signal."""
    return [
        130,
        128,
        126,
        124,
        122,
        120,
        118,
        116,
        114,
        112,
        110,
        108,
        106,
        104,
        102,
        100,
        98,
        96,
        94,
        92,
        90,
        88,
        86,
        84,
        82,
        80,
        78,
        76,
        74,
        72,
        85,
        95,
        105,
        115,
        125,
    ]


@pytest.fixture
def historical_prices_sell_signal() -> list[float]:
    """Provides a list of historical prices that will trigger a SELL signal."""
    return [
        70,
        72,
        74,
        76,
        78,
        80,
        82,
        84,
        86,
        88,
        90,
        92,
        94,
        96,
        98,
        100,
        102,
        104,
        106,
        108,
        110,
        112,
        114,
        116,
        118,
        120,
        122,
        124,
        126,
        128,
        115,
        105,
        95,
        85,
        75,
    ]


@pytest.fixture
def alphabot_input_data_factory(
    base_portfolio_state: PortfolioState,
) -> Callable[..., AlphaBotPayload]:
    """
    Provides an enhanced factory for creating AlphaBotPayload instances for AlphaBot tests.
    Allows overriding nested portfolio_state values.
    """

    def _input_data(**kwargs) -> AlphaBotPayload:
        portfolio_state_data = base_portfolio_state.model_dump()

        # Allow overriding nested portfolio_state keys for convenience
        if "portfolio_state" in kwargs:
            portfolio_state_data.update(kwargs.pop("portfolio_state"))

        # Create the final payload, allowing top-level overrides
        payload_data = {
            "historical_prices": [],
            "current_price": 100.0,
            "portfolio_state": PortfolioState(**portfolio_state_data),
            "day": 50,
            **kwargs,  # Apply any other top-level overrides
        }

        return AlphaBotPayload(**payload_data)

    return _input_data
