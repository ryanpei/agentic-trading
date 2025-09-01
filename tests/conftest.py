import pytest
import pytest_asyncio
from unittest.mock import MagicMock
from typing import Optional  # Import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService, Session
from google.adk.agents import BaseAgent  # Import BaseAgent for type hinting
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater


@pytest.fixture
def task_updater_fixture(mocker):
    """Provides a factory for creating real TaskUpdater instances with a new mocked EventQueue for each call."""

    def factory(task_id, context_id):
        # Create a new mock_event_queue for each TaskUpdater instance
        mock_event_queue = mocker.AsyncMock(spec=EventQueue)
        task_updater = TaskUpdater(
            event_queue=mock_event_queue, task_id=task_id, context_id=context_id
        )
        return task_updater, mock_event_queue

    return factory


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


from unittest.mock import patch


@pytest.fixture(params=["alphabot.agent_executor", "riskguard.agent_executor"])
def mock_runner(request, mocker):
    """Parameterized fixture to mock the Runner for different agents."""
    with patch(f"{request.param}.Runner") as mock:
        mock_runner_instance = mock.return_value
        mock_runner_instance.session_service = mocker.AsyncMock()
        mock_runner_instance.session_service.get_session = mocker.AsyncMock(
            return_value=None
        )
        mock_runner_instance.session_service.create_session = mocker.AsyncMock(
            return_value=mocker.MagicMock(id="test_session_id")
        )
        yield mock


async def adk_mock_alphabot_generator(final_state_delta, final_reason):
    """Mocks the async generator for AlphaBot."""
    yield MagicMock(
        actions=MagicMock(state_delta=final_state_delta),
        content=MagicMock(parts=[]),
        is_final_response=lambda: False,
    )
    yield MagicMock(
        actions=MagicMock(state_delta=None),
        content=MagicMock(parts=[MagicMock(text=final_reason)]),
        is_final_response=lambda: True,
    )


async def adk_mock_riskguard_generator(result_name, result_data):
    """Mocks the async generator for RiskGuard."""
    function_response_part = MagicMock()
    function_response_part.function_response.name = result_name
    function_response_part.function_response.response = result_data
    yield MagicMock(
        content=MagicMock(parts=[function_response_part]),
        is_final_response=lambda: True,
    )


async def run_executor_test(
    executor, request_message, mock_runner_instance, task_updater=None
):
    """Helper to run an agent executor's execute method for testing."""
    executor._adk_runner = mock_runner_instance
    request_obj = MagicMock()
    request_obj.message = request_message  # Directly assign the message object

    context = RequestContext(
        task_id="test-task-123",
        context_id="test-context-456",
        request=request_obj,
    )

    # If a real TaskUpdater is not passed, we don't need a real EventQueue
    event_queue = task_updater.event_queue if task_updater else EventQueue()

    # Patch the TaskUpdater lookup within the executor's execute method
    with patch("a2a.server.tasks.task_updater.TaskUpdater") as mock_task_updater_class:
        if task_updater:
            mock_task_updater_class.return_value = task_updater
        await executor.execute(context, event_queue)

    return context
