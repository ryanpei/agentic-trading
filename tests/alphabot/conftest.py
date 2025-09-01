import pytest
from typing import Callable, Dict, Any
from alphabot.agent import AlphaBotAgent


@pytest.fixture
def agent() -> AlphaBotAgent:
    """Provides an AlphaBotAgent instance."""
    return AlphaBotAgent(stock_ticker="TEST")


@pytest.fixture
def alphabot_input_data_factory(
    base_portfolio_state: dict,
) -> Callable[..., Dict[str, Any]]:
    """
    Provides an enhanced factory for creating input_data dictionaries for AlphaBot tests.
    Allows overriding nested portfolio_state values.
    """

    def _input_data(**kwargs) -> Dict[str, Any]:
        # Start with a base structure
        base_data = {
            "historical_prices": [],
            "current_price": 100.0,
            "portfolio_state": base_portfolio_state.copy(),
            "day": 50,
        }

        # Allow overriding top-level keys
        base_data.update(kwargs)

        # Allow overriding nested portfolio_state keys for convenience
        portfolio_overrides = kwargs.get("portfolio_state", {})
        base_data["portfolio_state"].update(portfolio_overrides)

        return base_data

    return _input_data
