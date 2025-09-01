import pytest
from pydantic import ValidationError
from typing import cast, Literal

from common.models import RiskCheckPayload, TradeProposal, PortfolioState
from common.config import DEFAULT_RISKGUARD_MAX_POS_SIZE


def test_risk_check_payload_valid():
    """Tests that a valid RiskCheckPayload can be created."""
    trade_proposal = TradeProposal(
        action="BUY", ticker="TECH", quantity=100, price=150.0
    )
    portfolio_state = PortfolioState(cash=50000.0, shares=200, total_value=80000.0)

    payload = RiskCheckPayload(
        trade_proposal=trade_proposal, portfolio_state=portfolio_state
    )

    assert payload.max_pos_size == DEFAULT_RISKGUARD_MAX_POS_SIZE
    assert payload.trade_proposal.action == "BUY"


def test_trade_proposal_invalid_action():
    """Tests that TradeProposal rejects an invalid action."""
    with pytest.raises(ValidationError) as exc_info:
        # Use cast to bypass static type checking for this invalid value test
        invalid_action = cast(Literal["BUY", "SELL"], "HOLD")
        TradeProposal(action=invalid_action, ticker="TECH", quantity=100, price=150.0)
    assert "Input should be 'BUY' or 'SELL'" in str(exc_info.value)


def test_risk_check_payload_missing_required_field():
    """Tests that Pydantic raises an error if required nested models are missing."""
    # Missing 'portfolio_state'
    with pytest.raises(ValidationError) as exc_info:
        invalid_data = {
            "trade_proposal": TradeProposal(
                action="SELL", ticker="TECH", quantity=50, price=155.0
            )
        }
        RiskCheckPayload(**invalid_data)  # type: ignore

    assert "portfolio_state" in str(exc_info.value)
