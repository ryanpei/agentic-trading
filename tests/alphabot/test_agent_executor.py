import pytest
from typing import Callable

from a2a.types import Message, DataPart, Part, Role
from alphabot.agent_executor import AlphaBotAgentExecutor
from tests.conftest import adk_mock_alphabot_generator, run_executor_test


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


@pytest.mark.asyncio
async def test_execute_success_buy_decision(
    alphabot_message_factory, mock_runner, task_updater_fixture
):
    mock_runner_instance = mock_runner.return_value
    task_updater, mock_event_queue = task_updater_fixture(
        "test-task-123", "test-context-456"
    )

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
    context = await run_executor_test(
        executor,
        request_message,
        mock_runner_instance,
        task_updater=task_updater,
    )

    # Assert
    assert mock_runner_instance.run_async.call_count == 1
    _, call_kwargs = mock_runner_instance.run_async.call_args
    assert call_kwargs["session_id"] == context.context_id

    # Verify events were enqueued correctly
    assert (
        mock_event_queue.enqueue_event.call_count == 4
    )  # submit, start, artifact, complete

    # More detailed assertions can be added here to inspect the enqueued events
    # For example, check the state of the TaskStatusUpdateEvent
    submit_call = mock_event_queue.enqueue_event.call_args_list[0]
    assert submit_call.args[0].status.state == "submitted"

    start_call = mock_event_queue.enqueue_event.call_args_list[1]
    assert start_call.args[0].status.state == "working"

    artifact_call = mock_event_queue.enqueue_event.call_args_list[2]
    expected_artifact_data = {
        "approved": True,
        "trade_proposal": {"action": "BUY", "quantity": 10},
        "reason": "SMA crossover indicates buy signal.",
    }
    assert artifact_call.args[0].artifact.parts[0].root.data == expected_artifact_data

    complete_call = mock_event_queue.enqueue_event.call_args_list[3]
    assert complete_call.args[0].status.state == "completed"


@pytest.mark.asyncio
async def test_execute_missing_market_data(mock_runner, task_updater_fixture):
    mock_runner_instance = mock_runner.return_value
    task_updater, mock_event_queue = task_updater_fixture(
        "test-task-123", "test-context-456"
    )

    # Arrange
    request_message = Message(
        message_id="test_message_id",
        role=Role.user,
        parts=[Part(root=DataPart(data={"cash": 10000.0, "shares": 100}))],
    )

    # Act
    executor = AlphaBotAgentExecutor()
    await run_executor_test(
        executor,
        request_message,
        mock_runner_instance,
        task_updater=task_updater,
    )

    # Assert
    assert mock_event_queue.enqueue_event.call_count == 3  # submit, start, failed
    submit_call = mock_event_queue.enqueue_event.call_args_list[0]
    assert submit_call.args[0].status.state == "submitted"

    start_call = mock_event_queue.enqueue_event.call_args_list[1]
    assert start_call.args[0].status.state == "working"

    fail_call = mock_event_queue.enqueue_event.call_args_list[2]
    assert fail_call.args[0].status.state == "failed"
    fail_message = fail_call.args[0].status.message
    assert "Invalid input: Missing market_data or portfolio_state" in str(
        fail_message.parts[0].root.data
    )


@pytest.mark.asyncio
async def test_execute_adk_runner_exception(
    alphabot_message_factory, mock_runner, task_updater_fixture
):
    mock_runner_instance = mock_runner.return_value
    task_updater, mock_event_queue = task_updater_fixture(
        "test-task-123", "test-context-456"
    )

    # Arrange
    mock_runner_instance.run_async.side_effect = Exception("ADK Borked")
    request_message = alphabot_message_factory(
        historical_prices=[150.0, 151.0, 152.0], current_price=155.0
    )

    # Act
    executor = AlphaBotAgentExecutor()
    await run_executor_test(
        executor,
        request_message,
        mock_runner_instance,
        task_updater=task_updater,
    )

    # Assert
    assert mock_event_queue.enqueue_event.call_count == 3  # submit, start, failed
    submit_call = mock_event_queue.enqueue_event.call_args_list[0]
    assert submit_call.args[0].status.state == "submitted"

    start_call = mock_event_queue.enqueue_event.call_args_list[1]
    assert start_call.args[0].status.state == "working"

    fail_call = mock_event_queue.enqueue_event.call_args_list[2]
    assert fail_call.args[0].status.state == "failed"
    fail_message = fail_call.args[0].status.message
    assert "ADK Agent error: ADK Borked" in str(fail_message.parts[0].root.data)
