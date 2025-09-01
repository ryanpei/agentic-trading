import pytest
from typing import Callable, Dict, Any
from common.config import (
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
)


@pytest.fixture
def riskguard_input_data_factory(
    base_trade_proposal: dict, base_portfolio_state: dict
) -> Callable[..., Dict[str, Any]]:
    """Provides a factory for creating input_data dictionaries for RiskGuard tests."""

    def _input_data(**kwargs) -> Dict[str, Any]:
        # Start with copies of the base fixtures to avoid modifying them directly
        current_trade_proposal = base_trade_proposal.copy()
        current_portfolio_state = base_portfolio_state.copy()

        # Merge any provided trade_proposal kwargs
        if "trade_proposal" in kwargs:
            current_trade_proposal.update(kwargs.pop("trade_proposal"))

        # Merge any provided portfolio_state kwargs
        if "portfolio_state" in kwargs:
            current_portfolio_state.update(kwargs.pop("portfolio_state"))

        base_data = {
            "trade_proposal": current_trade_proposal,
            "portfolio_state": current_portfolio_state,
            "max_pos_size": DEFAULT_RISKGUARD_MAX_POS_SIZE,
            "max_concentration": DEFAULT_RISKGUARD_MAX_CONCENTRATION,
        }
        base_data.update(kwargs)
        return base_data

    return _input_data
