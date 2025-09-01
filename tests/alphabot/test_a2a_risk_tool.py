import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from alphabot.a2a_risk_tool import A2ARiskCheckTool
from a2a.client import A2AClient
from a2a.types import (
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TaskStatus,
    Artifact,
    DataPart,
    TaskState,
    Part,
)
from google.adk.tools import ToolContext


@pytest.fixture
def mock_a2a_client():
    """Fixture to create a mock A2AClient."""
    client = MagicMock(spec=A2AClient)
    client.send_message = AsyncMock()
    return client


@pytest.fixture
def risk_check_tool():
    """Fixture to create an A2ARiskCheckTool."""
    return A2ARiskCheckTool()


@pytest.fixture
def tool_context():
    """Fixture to create a mock ToolContext."""
    mock_context = MagicMock(spec=ToolContext)
    mock_context.invocation_id = "test-invocation-id"
    # Configure the nested mock structure correctly
    mock_context._invocation_context = MagicMock()
    mock_context._invocation_context.session = MagicMock()
    mock_context._invocation_context.session.id = "test-session-id"
    return mock_context


def create_success_response(result_data: dict) -> SendMessageSuccessResponse:
    """Helper to create a successful A2A response for mocking."""
    return SendMessageSuccessResponse(
        id="request-id",
        jsonrpc="2.0",
        result=Task(
            id="task-id",
            context_id="test-session-id",
            status=TaskStatus(state=TaskState.completed),
            artifacts=[
                Artifact(
                    artifact_id="artifact-id",
                    name="risk_assessment",
                    parts=[Part(root=DataPart(data=result_data))],
                )
            ],
        ),
    )


@pytest.mark.asyncio
async def test_run_async_approved(
    risk_check_tool,
    tool_context,
    mock_a2a_client,
    base_trade_proposal,  # <-- Use fixture
    base_portfolio_state,  # <-- Use fixture
):
    """Test using shared fixtures for input data."""
    # Arrange
    # The base fixtures provide the data, reducing inline definitions
    args = {
        "trade_proposal": base_trade_proposal,
        "portfolio_state": base_portfolio_state,
    }

    expected_result = {"approved": True, "reason": "Within limits"}
    mock_response = create_success_response(expected_result)

    # We need to patch the A2AClient constructor within the run_async method
    with patch(
        "alphabot.a2a_risk_tool.A2AClient", return_value=mock_a2a_client
    ) as mock_client_constructor:
        mock_a2a_client.send_message.return_value = SendMessageResponse(
            root=mock_response
        )

        # Act
        events = [
            event async for event in risk_check_tool.run_async(args, tool_context)
        ]

        # Assert
        assert len(events) == 1
        response_part = events[0].content.parts[0]
        response_data = response_part.function_response.response
        assert response_data == expected_result

        # Verify the A2AClient was called correctly
        mock_client_constructor.assert_called_once()
        mock_a2a_client.send_message.assert_awaited_once()
        sent_request = mock_a2a_client.send_message.call_args[0][0]
        assert sent_request.params.message.metadata["max_pos_size"] is not None
        assert sent_request.params.message.metadata["max_concentration"] is not None


@pytest.mark.asyncio
async def test_run_async_rejected(
    risk_check_tool,
    tool_context,
    mock_a2a_client,
    base_trade_proposal,
    base_portfolio_state,
):
    """Test the tool's run_async method for a rejected trade."""
    # Arrange
    args = {
        "trade_proposal": base_trade_proposal,
        "portfolio_state": base_portfolio_state,
    }

    expected_result = {"approved": False, "reason": "Exceeds max position size"}
    mock_response = create_success_response(expected_result)

    with patch(
        "alphabot.a2a_risk_tool.A2AClient", return_value=mock_a2a_client
    ) as mock_client_constructor:
        mock_a2a_client.send_message.return_value = SendMessageResponse(
            root=mock_response
        )

        # Act
        events = [
            event async for event in risk_check_tool.run_async(args, tool_context)
        ]

        # Assert
        assert len(events) == 1
        response_part = events[0].content.parts[0]
        response_data = response_part.function_response.response
        assert response_data == expected_result

        mock_client_constructor.assert_called_once()
