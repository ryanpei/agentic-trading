import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from typing import Optional  # Import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService, Session
from google.adk.agents import BaseAgent  # Import BaseAgent for type hinting
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue


@pytest_asyncio.fixture
async def adk_session() -> Session:
    """Provides a reusable, awaited session instance for tests."""
    return await InMemorySessionService().create_session(
        app_name="test_app", user_id="test_user"
    )


@pytest.fixture
def adk_ctx(
    adk_session: Session, agent: Optional[BaseAgent] = None
) -> InvocationContext:  # Use Optional
    """Provides a base InvocationContext."""
    # If no agent is provided, use a MagicMock to satisfy the BaseAgent type hint
    if agent is None:
        agent = MagicMock(spec=BaseAgent)
        agent.name = "MockAgent"  # Assign a name to the mock agent

    return InvocationContext(
        agent=agent,
        session_service=InMemorySessionService(),
        invocation_id="test_invocation",
        session=adk_session,
    )


@pytest.fixture
def base_portfolio_state() -> dict:
    """Provides a default portfolio state, shared across all tests."""
    return {"cash": 100000.0, "shares": 100, "total_value": 110000.0}


@pytest.fixture
def base_trade_proposal() -> dict:
    """Provides a default 'BUY' trade proposal, shared across all tests."""
    return {
        "action": "BUY",
        "ticker": "TECH",
        "quantity": 50,
        "price": 100.0,
    }


@pytest.fixture
def mock_runner_factory(mocker):
    """Factory fixture to create a mock Runner for a specific agent."""

    def _factory(agent_module_path: str):
        with patch(f"{agent_module_path}.Runner") as mock:
            mock_runner_instance = mock.return_value
            mock_runner_instance.session_service = mocker.AsyncMock()
            mock_runner_instance.session_service.get_session = mocker.AsyncMock(
                return_value=None
            )
            mock_runner_instance.session_service.create_session = mocker.AsyncMock(
                return_value=mocker.MagicMock(id="test_session_id")
            )
            return mock_runner_instance

    return _factory


@pytest.fixture
def adk_mock_alphabot_generator():
    """
    Mocks the async generator for AlphaBot, yielding ADK Events instead of
    raw genai types. This aligns with the behavior of the ADK Runner.
    """

    async def _generator(final_state_delta, final_reason):
        # Yield an event with the state delta
        yield Event(
            author="test_author",
            actions=EventActions(state_delta=final_state_delta)
        )
        # Yield a final event with the reason text
        yield Event(
            author="test_author",
            content=genai_types.Content(parts=[genai_types.Part(text=final_reason)]),
            turn_complete=True,
        )

    return _generator


@pytest.fixture
def adk_mock_riskguard_generator():
    """
    Mocks the async generator for RiskGuard, yielding a single ADK Event
    with a function response. This aligns with the behavior of the ADK Runner.
    """

    async def _generator(result_name, result_data):
        yield Event(
            author="test_author",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=result_name, response=result_data
                        )
                    )
                ]
            ),
            turn_complete=True,
        )

    return _generator
