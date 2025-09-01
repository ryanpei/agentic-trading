import pytest
from alphabot.agent import AlphaBotAgent
from common.models import AlphaBotTaskPayload, PortfolioState
from common.utils.agent_utils import create_a2a_message_from_payload
from a2a.types import Message, Role


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
def alphabot_input_data_factory(base_portfolio_state: PortfolioState):
    """
    Provides a factory for creating AlphaBotTaskPayload instances.
    Injects shared base model fixtures from the root conftest.
    """

    def _input_data(**kwargs) -> AlphaBotTaskPayload:
        # Use the same .model_copy() pattern as the refactored riskguard factory
        final_portfolio_state = base_portfolio_state.model_copy(deep=True)

        if "portfolio_state" in kwargs:
            final_portfolio_state = PortfolioState(**kwargs.pop("portfolio_state"))

        payload = AlphaBotTaskPayload(portfolio_state=final_portfolio_state, **kwargs)
        return payload

    return _input_data


@pytest.fixture
def alphabot_message_factory(alphabot_input_data_factory):
    """Factory to create a complete A2A Message for AlphaBot tests using the common utility."""

    def _create_message(**kwargs) -> Message:
        # 1. Create the payload using the other factory
        input_data = alphabot_input_data_factory(**kwargs)

        # 2. Use the existing utility to create the message
        # This ensures the test uses the same logic as the application
        a2a_message = create_a2a_message_from_payload(input_data, role=Role.user)

        # You can still override generated fields if needed for specific tests
        a2a_message.context_id = kwargs.get("context_id", a2a_message.context_id)

        return a2a_message

    return _create_message
