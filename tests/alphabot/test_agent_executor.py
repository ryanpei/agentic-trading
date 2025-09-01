from typing import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.types import DataPart, Message, Part, Role
from alphabot.agent_executor import AlphaBotAgentExecutor


@pytest.fixture
def alphabot_message_factory(
    alphabot_input_data_factory,
) -> Callable[..., Message]:
    """Factory to create a complete A2A Message for AlphaBot tests."""

    def _create_message(**kwargs) -> Message:
        input_data = alphabot_input_data_factory(**kwargs)
        # The AlphaBot executor expects two separate parts
        market_data = {
            "historical_prices": input_data["historical_prices"],
            "current_price": input_data["current_price"],
            "day": input_data["day"],
        }
        portfolio_state = input_data["portfolio_state"]
        return Message(
            message_id="test_message_id",
            role=Role.user,
            parts=[
                Part(root=DataPart(data=market_data)),
                Part(root=DataPart(data=portfolio_state)),
            ],
        )

    return _create_message


@pytest.fixture
def mock_event_queue():
    """Mock EventQueue for testing."""
    return AsyncMock()


@pytest.mark.asyncio
async def test_execute_success_buy_decision(
    alphabot_message_factory,
    mock_runner_factory,
    mock_event_queue,
    adk_mock_alphabot_generator,
):
    mock_runner_instance = mock_runner_factory("alphabot.agent_executor")

    # Arrange
    request_message = alphabot_message_factory(
        historical_prices=[150.0, 151.0, 152.0], current_price=155.0, day=1
    )
    mock_runner_instance.run_async.return_value = adk_mock_alphabot_generator(
        final_state_delta={"approved_trade": {"action": "BUY", "quantity": 10}},
        final_reason="SMA crossover indicates buy signal.",
    )

    # Act
    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner_instance  # Inject the mock runner
    await executor.execute(
        context=MagicMock(
            message=request_message,
            context_id="test-context-456",
            task_id="test-task-123",
        ),
        event_queue=mock_event_queue,
    )

    # Assert
    assert mock_runner_instance.run_async.call_count == 1
    _, call_kwargs = mock_runner_instance.run_async.call_args
    assert call_kwargs["session_id"] == "test-context-456"

    # Verify a single message was enqueued
    mock_event_queue.enqueue_event.assert_called_once()
    enqueued_message = mock_event_queue.enqueue_event.call_args[0][0]

    assert isinstance(enqueued_message, Message)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    assert len(enqueued_message.parts) == 1
    assert isinstance(enqueued_message.parts[0].root, DataPart)

    expected_data = {
        "approved": True,
        "trade_proposal": {"action": "BUY", "quantity": 10},
        "reason": "SMA crossover indicates buy signal.",
    }
    assert enqueued_message.parts[0].root.data == expected_data
    mock_event_queue.close.assert_called_once()


@pytest.mark.asyncio
async def test_execute_missing_market_data(mock_runner_factory, mock_event_queue):
    mock_runner_instance = mock_runner_factory("alphabot.agent_executor")

    # Arrange
    request_message = Message(
        message_id="test_message_id",
        role=Role.user,
        parts=[Part(root=DataPart(data={"cash": 10000.0, "shares": 100}))],
    )

    # Act
    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner_instance  # Inject the mock runner
    await executor.execute(
        context=MagicMock(
            message=request_message,
            context_id="test-context-456",
            task_id="test-task-123",
        ),
        event_queue=mock_event_queue,
    )

    # Assert
    mock_event_queue.enqueue_event.assert_called_once()
    enqueued_message = mock_event_queue.enqueue_event.call_args[0][0]

    assert isinstance(enqueued_message, Message)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    assert len(enqueued_message.parts) == 1
    assert isinstance(enqueued_message.parts[0].root, DataPart)
    assert (
        "Invalid input: Missing market_data or portfolio_state"
        in enqueued_message.parts[0].root.data["error"]
    )
    mock_event_queue.close.assert_called_once()


@pytest.mark.asyncio
async def test_execute_adk_runner_exception(
    alphabot_message_factory,
    mock_runner_factory,
    mock_event_queue,
    adk_mock_alphabot_generator,
):
    mock_runner_instance = mock_runner_factory("alphabot.agent_executor")

    # Arrange
    mock_runner_instance.run_async.side_effect = Exception("ADK Borked")
    request_message = alphabot_message_factory(
        historical_prices=[150.0, 151.0, 152.0], current_price=155.0
    )

    # Act
    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner_instance  # Inject the mock runner
    await executor.execute(
        context=MagicMock(
            message=request_message,
            context_id="test-context-456",
            task_id="test-task-123",
        ),
        event_queue=mock_event_queue,
    )

    # Assert
    mock_event_queue.enqueue_event.assert_called_once()
    enqueued_message = mock_event_queue.enqueue_event.call_args[0][0]

    assert isinstance(enqueued_message, Message)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    assert len(enqueued_message.parts) == 1
    assert isinstance(enqueued_message.parts[0].root, DataPart)
    assert "ADK Agent error: ADK Borked" in enqueued_message.parts[0].root.data["error"]
    mock_event_queue.close.assert_called_once()


@pytest.mark.asyncio
async def test_execute_handles_adk_runner_exception(
    alphabot_message_factory,
    mock_runner_factory,
    mock_event_queue,
    adk_mock_alphabot_generator,
):
    """
    Tests that if the ADK runner fails, the executor enqueues an error message
    and closes the queue.
    """
    # Arrange
    mock_runner = mock_runner_factory("alphabot.agent_executor")
    # Simulate an exception during the ADK agent's execution
    mock_runner.run_async.side_effect = Exception("ADK agent failed!")

    request_message = alphabot_message_factory()
    context = MagicMock(
        message=request_message, context_id="test-context-456", task_id="test-task-123"
    )

    # Act
    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner  # Inject the mock runner

    await executor.execute(context, mock_event_queue)

    # Assert
    # 1. An event was enqueued
    mock_event_queue.enqueue_event.assert_called_once()

    # 2. The enqueued event is a Message containing error details
    enqueued_message = mock_event_queue.enqueue_event.call_args[0][0]
    assert isinstance(enqueued_message, Message)

    # 3. The message part contains the error
    error_part = enqueued_message.parts[0].root
    assert isinstance(error_part, DataPart)
    assert error_part.data["error"] is not None
    assert "ADK Agent error: ADK agent failed!" in error_part.data["error"]

    # 4. The queue was closed (critical for preventing hangs)
    mock_event_queue.close.assert_called_once()
