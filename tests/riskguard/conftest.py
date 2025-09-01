import pytest
from common.models import (
    PortfolioState,
    TradeProposal,
    RiskCheckPayload,
)


@pytest.fixture
def riskguard_input_data_factory(
    base_trade_proposal: TradeProposal, base_portfolio_state: PortfolioState
):
    """
    Provides a factory for creating RiskCheckPayload instances for RiskGuard tests.
    Injects shared base model fixtures from the root conftest.
    """

    def _input_data(**kwargs) -> RiskCheckPayload:
        # 1. Use the injected fixtures as the starting point
        final_trade_proposal = base_trade_proposal.model_copy(deep=True)
        final_portfolio_state = base_portfolio_state.model_copy(deep=True)

        # 2. Update models directly from kwargs if needed
        if "trade_proposal" in kwargs:
            final_trade_proposal = TradeProposal(**kwargs.pop("trade_proposal"))
        if "portfolio_state" in kwargs:
            final_portfolio_state = PortfolioState(**kwargs.pop("portfolio_state"))

        # 3. Create the final payload from the model instances
        payload = RiskCheckPayload(
            trade_proposal=final_trade_proposal,
            portfolio_state=final_portfolio_state,
            **kwargs,  # Apply any other top-level overrides
        )
        return payload

    return _input_data
