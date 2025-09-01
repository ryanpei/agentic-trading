import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from alphabot.a2a_risk_tool import A2ARiskCheckTool
from alphabot.agent import AlphaBotAgent
from a2a.client import A2AClient
from a2a.types import (
    SendMessageResponse,
    SendMessageSuccessResponse,
    DataPart,
    Part,
    Message,
    Role,
)
from google.adk.tools import ToolContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import Session
from google.adk.sessions.in_memory_session_service import InMemorySessionService


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
def tool_context(adk_session: Session):
    """Fixture to create a mock ToolContext."""
    return ToolContext(
        invocation_context=InvocationContext(
            invocation_id="test-invocation-id",
            session=adk_session,
            agent=AlphaBotAgent(),
            session_service=InMemorySessionService(),
        )
    )


def create_success_response(result_data: dict) -> SendMessageSuccessResponse:
    """Helper to create a successful A2A response for mocking."""
    return SendMessageSuccessResponse(
        id="request-id",
        jsonrpc="2.0",
        result=Message(
            message_id="test-message-id",
            context_id="test-context-id",
            task_id="test-task-id",
            role=Role.agent,
            parts=[Part(root=DataPart(data=result_data))],
        ),
    )


def _verify_a2a_call(
    mock_a2a_client: MagicMock, expected_trade_proposal: dict, expected_portfolio_state: dict
):
    """Helper function to verify the payload sent to the A2AClient."""
    mock_a2a_client.send_message.assert_awaited_once()
    sent_request = mock_a2a_client.send_message.call_args[0][0]
    sent_message: Message = sent_request.params.message
    sent_part = sent_message.parts[0].root
    assert isinstance(sent_part, DataPart)
    sent_payload = sent_part.data
    assert isinstance(sent_payload, dict)
    assert sent_payload["trade_proposal"] == expected_trade_proposal
    assert sent_payload["portfolio_state"] == expected_portfolio_state


@pytest.mark.asyncio
async def test_run_async_approved(
    risk_check_tool,
    tool_context,
    mock_a2a_client,
    base_trade_proposal,
    base_portfolio_state,
):
    """Test using shared fixtures for input data."""
    # Arrange
    args = {
        "trade_proposal": base_trade_proposal,
        "portfolio_state": base_portfolio_state,
    }
    expected_result = {"approved": True, "reason": "Within limits"}
    mock_response = create_success_response(expected_result)

    with patch(
        "alphabot.a2a_risk_tool.A2AClient", return_value=mock_a2a_client
    ) as mock_client_constructor:
        mock_a2a_client.send_message.return_value = SendMessageResponse(
            root=mock_response
        )

        # Act
        events = [
            event async for event in risk_check_tool.run_async(args=args, tool_context=tool_context)
        ]

        # Assert
        assert len(events) == 1
        response_part = events[0].content.parts[0]
        response_data = response_part.function_response.response
        assert response_data == expected_result

        mock_client_constructor.assert_called_once()
        _verify_a2a_call(
            mock_a2a_client, base_trade_proposal, base_portfolio_state
        )


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
            event async for event in risk_check_tool.run_async(args=args, tool_context=tool_context)
        ]

        # Assert
        assert len(events) == 1
        response_part = events[0].content.parts[0]
        response_data = response_part.function_response.response
        assert response_data == expected_result

        mock_client_constructor.assert_called_once()
        _verify_a2a_call(
            mock_a2a_client, base_trade_proposal, base_portfolio_state
        )
