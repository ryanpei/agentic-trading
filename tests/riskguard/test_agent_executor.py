import pytest
from typing import Callable

from a2a.types import Message, DataPart, Part, Role
from riskguard.agent_executor import RiskGuardAgentExecutor
from tests.conftest import adk_mock_riskguard_generator, run_executor_test


@pytest.fixture
def riskguard_message_factory(
    riskguard_input_data_factory,
) -> Callable[..., Message]:
    """Factory to create a complete A2A Message for RiskGuard tests."""

    def _create_message(**kwargs) -> Message:
        input_data = riskguard_input_data_factory(**kwargs)
        return Message(
            message_id="test_message_id",
            role=Role.user,
            parts=[Part(root=DataPart(data=input_data))],
        )

    return _create_message


@pytest.mark.asyncio
async def test_execute_success_approved(
    riskguard_message_factory, mock_runner, task_updater_fixture
):
    """Test with a simplified message creation."""
    mock_runner_instance = mock_runner.return_value
    task_updater, mock_event_queue = task_updater_fixture(
        "test-task-123", "test-context-456"
    )

    # Arrange
    request_message = riskguard_message_factory(
        trade_proposal={"quantity": 10}, portfolio_state={"cash": 50000}
    )
    mock_runner_instance.run_async.return_value = adk_mock_riskguard_generator(
        result_name="risk_check_result",
        result_data={"approved": True, "reason": "Within risk parameters."},
    )

    # Act
    executor = RiskGuardAgentExecutor()
    await run_executor_test(
        executor,
        request_message,
        mock_runner_instance,
        task_updater=task_updater,
    )

    # Assert
    assert mock_runner_instance.run_async.call_count == 1
    assert (
        mock_event_queue.enqueue_event.call_count == 4
    )  # submit, start, artifact, complete

    submit_call = mock_event_queue.enqueue_event.call_args_list[0]
    assert submit_call.args[0].status.state == "submitted"

    start_call = mock_event_queue.enqueue_event.call_args_list[1]
    assert start_call.args[0].status.state == "working"

    artifact_call = mock_event_queue.enqueue_event.call_args_list[2]
    expected_artifact = {"approved": True, "reason": "Within risk parameters."}
    assert artifact_call.args[0].artifact.parts[0].root.data == expected_artifact

    complete_call = mock_event_queue.enqueue_event.call_args_list[3]
    assert complete_call.args[0].status.state == "completed"


@pytest.mark.asyncio
async def test_execute_missing_trade_proposal(mock_runner, task_updater_fixture):
    mock_runner_instance = mock_runner.return_value
    task_updater, mock_event_queue = task_updater_fixture(
        "test-task-123", "test-context-456"
    )

    # Arrange
    request_message = Message(
        message_id="test_message_id",
        role=Role.user,
        parts=[
            Part(
                root=DataPart(
                    data={
                        "portfolio_state": {
                            "cash": 10000.0,
                            "shares": 100,
                            "total_value": 20000.0,
                        }
                    }
                )
            )
        ],
    )

    # Act
    executor = RiskGuardAgentExecutor()
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
    assert "Invalid input data" in str(fail_message.parts[0].root.data)
    mock_runner_instance.run_async.assert_not_called()


@pytest.mark.asyncio
async def test_execute_adk_runner_exception(
    riskguard_message_factory, mock_runner, task_updater_fixture
):
    mock_runner_instance = mock_runner.return_value
    task_updater, mock_event_queue = task_updater_fixture(
        "test-task-123", "test-context-456"
    )

    # Arrange
    mock_runner_instance.run_async.side_effect = Exception("ADK Borked")
    request_message = riskguard_message_factory()

    # Act
    executor = RiskGuardAgentExecutor()
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
