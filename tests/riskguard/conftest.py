import pytest
from typing import Callable
from common.models import (
    PortfolioState,
    TradeProposal,
    RiskCheckPayload,
)


@pytest.fixture
def riskguard_input_data_factory(
    base_trade_proposal: TradeProposal, base_portfolio_state: PortfolioState
) -> Callable[..., RiskCheckPayload]:
    """
    Provides a factory for creating RiskCheckPayload instances for RiskGuard tests.
    Allows overriding of nested models and their fields.
    """

    def _input_data(**kwargs) -> RiskCheckPayload:
        trade_proposal_data = base_trade_proposal.model_dump()
        portfolio_state_data = base_portfolio_state.model_dump()

        # If trade_proposal or portfolio_state are passed as kwargs,
        # update the base data with their fields.
        if "trade_proposal" in kwargs:
            trade_proposal_data.update(kwargs.pop("trade_proposal"))
        if "portfolio_state" in kwargs:
            portfolio_state_data.update(kwargs.pop("portfolio_state"))

        # Create the final payload, allowing top-level overrides
        payload_data = {
            "trade_proposal": TradeProposal(**trade_proposal_data),
            "portfolio_state": PortfolioState(**portfolio_state_data),
            **kwargs,  # Apply any other top-level overrides
        }

        return RiskCheckPayload(**payload_data)

    return _input_data
