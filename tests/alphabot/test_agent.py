import pytest
import json
from unittest.mock import MagicMock, AsyncMock

# ADK Imports
try:
    from google.adk.agents.invocation_context import InvocationContext
    from google.genai import types as genai_types
    from google.adk.events import Event
    from google.adk.sessions import InMemorySessionService
except ImportError:
    from tests.adk_mocks import (
        InvocationContext, genai_types, Event, InMemorySessionService
    )


from alphabot.agent import AlphaBotAgent, A2ARiskCheckTool
from common.config import DEFAULT_TICKER

def test_alphabot_agent_instantiation():
    """Tests basic instantiation of the AlphaBotAgent."""
    try:
        agent = AlphaBotAgent(stock_ticker="TEST_TICKER")
        assert agent is not None
        assert agent.name == "AlphaBot"
        assert agent.ticker == "TEST_TICKER"
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], A2ARiskCheckTool)
        default_agent = AlphaBotAgent()
        assert default_agent.ticker == DEFAULT_TICKER
    except Exception as e:
        pytest.fail(f"AlphaBotAgent instantiation failed: {e}")


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_no_signal():
    """Tests _run_async_impl when no crossover signal is generated."""
    agent = AlphaBotAgent(stock_ticker="TEST")
    agent._should_be_long = False

    mock_a2a_tool = AsyncMock(spec=A2ARiskCheckTool)
    mock_a2a_tool.name = "MockTool"
    async def mock_run_async(*args, **kwargs):
        yield Event(author=mock_a2a_tool.name, content=MagicMock())
        if False: # pragma: no cover
             yield
    mock_a2a_tool.run_async = mock_run_async
    agent.tools = [mock_a2a_tool]

    input_data = {
        "historical_prices": [100, 101, 102, 103, 104, 105],
        "current_price": 105.5,
        "portfolio_state": {"cash": 10000, "shares": 0, "total_value": 10000},
        "short_sma_period": 2,
        "long_sma_period": 4,
        "trade_quantity": 10,
        "riskguard_url": "mock_url",
        "max_pos_size": 5000,
        "max_concentration": 0.5
    }
    mock_content = genai_types.Content(parts=[genai_types.Part(text=json.dumps(input_data))])
    mock_session_service = InMemorySessionService()
    mock_session = mock_session_service.create_session(app_name="test_app", user_id="test_user")
    mock_ctx = InvocationContext(
        user_content=mock_content,
        session_service=mock_session_service,
        invocation_id="test_invocation_1",
        agent=agent,
        session=mock_session
    )

    events = []
    async for event in agent._run_async_impl(mock_ctx):
        events.append(event)

    assert len(events) == 1
    assert events[0].author == agent.name
    assert "No signal" in events[0].content.parts[0].text
    assert not events[0].actions or not events[0].actions.state_delta
