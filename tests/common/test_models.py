import pytest
from pydantic import ValidationError
from common.models import (
    TradeProposal,
    PortfolioState,
    RiskCheckPayload,
    AlphaBotPayload,
)


def test_trade_proposal_valid():
    data = {"action": "BUY", "ticker": "TEST", "quantity": 10, "price": 150.0}
    proposal = TradeProposal(**data)
    assert proposal.action == "BUY"
    assert proposal.ticker == "TEST"
    assert proposal.quantity == 10
    assert proposal.price == 150.0


def test_trade_proposal_invalid():
    with pytest.raises(ValidationError):
        invalid_data = {
            "action": "BUY",
            "ticker": "TEST",
            "quantity": "ten",
            "price": 150.0,
        }
        TradeProposal(**invalid_data)


def test_portfolio_state_valid():
    data = {"cash": 10000.0, "shares": 50, "total_value": 17500.0}
    portfolio = PortfolioState(**data)
    assert portfolio.cash == 10000.0
    assert portfolio.shares == 50
    assert portfolio.total_value == 17500.0


def test_risk_check_payload_valid():
    trade_proposal = TradeProposal(
        action="BUY", ticker="TEST", quantity=10, price=150.0
    )
    portfolio_state = PortfolioState(cash=10000.0, shares=50, total_value=17500.0)
    payload = RiskCheckPayload(
        trade_proposal=trade_proposal, portfolio_state=portfolio_state
    )
    assert payload.trade_proposal == trade_proposal
    assert payload.portfolio_state == portfolio_state


def test_alphabot_payload_valid():
    portfolio_state = PortfolioState(cash=10000.0, shares=50, total_value=17500.0)
    payload = AlphaBotPayload(
        historical_prices=[140.0, 145.0, 150.0],
        current_price=155.0,
        portfolio_state=portfolio_state,
        day=1,
    )
    assert payload.current_price == 155.0
    assert payload.portfolio_state == portfolio_state
