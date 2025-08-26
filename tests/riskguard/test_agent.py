import json
import pytest
import pytest_asyncio

# ADK Imports
try:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.sessions import InMemorySessionService, Session
    from google.genai import types as genai_types
except ImportError:
    from tests.adk_mocks import (
        InvocationContext,
        Session,
        genai_types,
        InMemorySessionService,
    )

from common.config import (
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
)
from riskguard.agent import RiskGuardAgent


@pytest.fixture
def agent() -> RiskGuardAgent:
    """Provides a RiskGuardAgent instance."""
    return RiskGuardAgent()


@pytest_asyncio.fixture
async def mock_session() -> Session:
    """Provides an awaited session instance."""
    return await InMemorySessionService().create_session(
        app_name="test_app", user_id="test_user"
    )


@pytest.fixture
def mock_ctx(agent: RiskGuardAgent, mock_session: Session) -> InvocationContext:
    """Provides a base InvocationContext."""
    return InvocationContext(
        agent=agent,
        session_service=InMemorySessionService(),
        invocation_id="test_invocation",
        session=mock_session,
    )


def test_riskguard_agent_instantiation(agent: RiskGuardAgent):
    """Tests basic instantiation of the RiskGuardAgent."""
    assert agent is not None
    assert agent.name == "RiskGuard"
    assert (
        agent.description == "Evaluates proposed trades against predefined risk rules."
    )


@pytest.mark.asyncio
async def test_riskguard_run_async_impl_approve(agent, mock_ctx: InvocationContext):
    """Tests _run_async_impl approves a valid trade."""
    ctx = mock_ctx
    input_data = {
        "trade_proposal": {
            "action": "BUY",
            "ticker": "TEST",
            "quantity": 10,
            "price": 100.0,
        },
        "portfolio_state": {
            "cash": 10000,
            "shares": 0,
            "total_value": 10000,
            "positions": {},
        },
        "max_pos_size": DEFAULT_RISKGUARD_MAX_POS_SIZE,
        "max_concentration": DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    }
    ctx.user_content = genai_types.Content(
        parts=[genai_types.Part(text=json.dumps(input_data))]
    )

    events = []
    async for event in agent._run_async_impl(ctx):
        events.append(event)

    assert len(events) == 1
    final_event = events[0]
    assert final_event.author == agent.name
    assert final_event.turn_complete is True
    assert final_event.content.parts[0].function_response is not None

    result_data = final_event.content.parts[0].function_response.response
    assert result_data["approved"] is True
    assert result_data["reason"] == "Trade adheres to risk rules."


@pytest.mark.asyncio
async def test_riskguard_run_async_impl_reject_pos_size(
    agent, mock_ctx: InvocationContext
):
    """Tests _run_async_impl rejects a trade exceeding max position size."""
    ctx = mock_ctx
    input_data = {
        "trade_proposal": {
            "action": "BUY",
            "ticker": "TEST",
            "quantity": 60,
            "price": 100.0,
        },
        "portfolio_state": {
            "cash": 10000,
            "shares": 0,
            "total_value": 10000,
            "positions": {},
        },
        "max_pos_size": 5000,
        "max_concentration": 0.8,
    }
    ctx.user_content = genai_types.Content(
        parts=[genai_types.Part(text=json.dumps(input_data))]
    )

    events = []
    async for event in agent._run_async_impl(ctx):
        events.append(event)

    assert len(events) == 1
    final_event = events[0]
    assert final_event.author == agent.name
    assert final_event.turn_complete is True
    assert final_event.content.parts[0].function_response is not None

    result_data = final_event.content.parts[0].function_response.response
    assert result_data["approved"] is False
    assert "Exceeds max position size per trade" in result_data["reason"]
