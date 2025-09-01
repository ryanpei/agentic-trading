# tests/common/test_models.py

import pytest
from pydantic import ValidationError

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
    with pytest.raises(ValidationError):
        TradeProposal(action="HOLD", ticker="TECH", quantity=100, price=150.0)  # type: ignore


def test_risk_check_payload_missing_required_field():
    """Tests that Pydantic raises an error if required nested models are missing."""
    # Missing 'portfolio_state'
    with pytest.raises(ValidationError) as exc_info:
        RiskCheckPayload(  # type: ignore
            trade_proposal=TradeProposal(
                action="SELL", ticker="TECH", quantity=50, price=155.0
            )
        )

    assert "portfolio_state" in str(exc_info.value)
