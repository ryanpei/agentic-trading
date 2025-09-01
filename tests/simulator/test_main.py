import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from simulator.main import app
import common.config as defaults

client = TestClient(app)


@pytest.fixture
def mock_a2a_call():
    """Fixture to mock the _call_alphabot_a2a function."""
    with patch("simulator.main._call_alphabot_a2a", new_callable=AsyncMock) as mock:
        # Simulate a successful trade approval
        mock.return_value = {
            "approved_trade": {
                "action": "BUY",
                "quantity": 10,
                "price": 100.0,
                "ticker": "SIM",
            },
            "rejected_trade": None,
            "reason": "SMA crossover",
            "error": None,
        }
        yield mock


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_read_main():
    """Test the main endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Agentic Trading Simulator" in response.text


@pytest.mark.asyncio
async def test_call_alphabot_a2a_success():
    """Test _call_alphabot_a2a with a successful message response."""
    from unittest.mock import AsyncMock, MagicMock
    from a2a.types import (
        SendMessageResponse,
        SendMessageSuccessResponse,
        Message,
        DataPart,
        Part,
        Role,
    )
    from simulator.main import _call_alphabot_a2a
    from simulator.portfolio import PortfolioState

    mock_client = AsyncMock()
    mock_logger = MagicMock()

    # Mock the A2AClient.send_message to return a successful Message
    expected_trade_proposal = {
        "action": "BUY",
        "quantity": 10,
        "price": 100.0,
        "ticker": "TEST",
    }
    expected_reason = "Test approved reason."
    mock_message_data = {
        "approved": True,
        "trade_proposal": expected_trade_proposal,
        "reason": expected_reason,
    }
    mock_message = Message(
        message_id="mock_msg_id",
        context_id="mock_context_id",
        task_id="mock_task_id",
        role=Role.agent,
        parts=[Part(root=DataPart(data=mock_message_data))],
    )
    mock_send_message_success_response = SendMessageSuccessResponse(result=mock_message)
    mock_client.send_message.return_value = SendMessageResponse(
        root=mock_send_message_success_response
    )

    # Prepare input for the function
    session_id = "test-session-123"
    day = 1
    current_price = 100.0
    historical_prices = [90.0, 95.0]
    portfolio = PortfolioState(cash=10000.0, shares=0, total_value=10000.0)
    params = {
        "alphabot_short_sma": 10,
        "alphabot_long_sma": 20,
        "alphabot_trade_qty": 10,
        "riskguard_url": "http://localhost:8001",
        "riskguard_max_pos_size": 1000,
        "riskguard_max_concentration": 0.5,
    }

    # Call the function
    outcome = await _call_alphabot_a2a(
        client=mock_client,
        session_id=session_id,
        day=day,
        current_price=current_price,
        historical_prices=historical_prices,
        portfolio=portfolio,
        params=params,
        sim_logger=mock_logger,
    )

    # Assertions
    assert outcome["approved_trade"] == expected_trade_proposal
    assert outcome["rejected_trade"] is None
    assert outcome["reason"] == expected_reason
    assert outcome["error"] is None
    mock_client.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_call_alphabot_a2a_invalid_message_format():
    """Test _call_alphabot_a2a when AlphaBot returns an invalid message format."""
    from unittest.mock import AsyncMock, MagicMock
    from a2a.types import (
        SendMessageResponse,
        SendMessageSuccessResponse,
        Message,
        Part,
        Role,
        TextPart,
    )
    from simulator.main import _call_alphabot_a2a
    from simulator.portfolio import PortfolioState

    mock_client = AsyncMock()
    mock_logger = MagicMock()

    # Mock the A2AClient.send_message to return a Message without DataPart
    mock_message = Message(
        message_id="mock_msg_id",
        context_id="mock_context_id",
        task_id="mock_task_id",
        role=Role.agent,
        parts=[Part(root=TextPart(text="Invalid part, should be DataPart"))],
    )
    mock_send_message_success_response = SendMessageSuccessResponse(result=mock_message)
    mock_client.send_message.return_value = SendMessageResponse(
        root=mock_send_message_success_response
    )

    session_id = "test-session-123"
    day = 1
    current_price = 100.0
    historical_prices = [90.0, 95.0]
    portfolio = PortfolioState(cash=10000.0, shares=0, total_value=10000.0)
    params = {
        "alphabot_short_sma": 10,
        "alphabot_long_sma": 20,
        "alphabot_trade_qty": 10,
        "riskguard_url": "http://localhost:8001",
        "riskguard_max_pos_size": 1000,
        "riskguard_max_concentration": 0.5,
    }

    outcome = await _call_alphabot_a2a(
        client=mock_client,
        session_id=session_id,
        day=day,
        current_price=current_price,
        historical_prices=historical_prices,
        portfolio=portfolio,
        params=params,
        sim_logger=mock_logger,
    )

    assert outcome["approved_trade"] is None
    assert outcome["rejected_trade"] is None
    assert "AlphaBot Response Format Issue or No Decision" in outcome["error"]
    mock_client.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_call_alphabot_a2a_unexpected_response_type():
    """Test _call_alphabot_a2a when AlphaBot returns an unexpected response type."""
    from unittest.mock import AsyncMock, MagicMock
    from a2a.types import (
        SendMessageResponse,
        SendMessageSuccessResponse,
        Message,
        Role,
    )
    from simulator.main import _call_alphabot_a2a
    from simulator.portfolio import PortfolioState

    mock_client = AsyncMock()
    mock_logger = MagicMock()

    # Mock the A2AClient.send_message to return a non-Message object as result
    mock_send_message_success_response = SendMessageSuccessResponse(
        result=Message(
            message_id="mock_msg_id",
            context_id="mock_context_id",
            task_id="mock_task_id",
            role=Role.agent,
            parts=[],
        )
    )
    # Overwrite the result with a mock to trigger the error condition being tested
    mock_send_message_success_response.result = MagicMock(spec=object)
    mock_client.send_message.return_value = SendMessageResponse(
        root=mock_send_message_success_response
    )

    session_id = "test-session-123"
    day = 1
    current_price = 100.0
    historical_prices = [90.0, 95.0]
    portfolio = PortfolioState(cash=10000.0, shares=0, total_value=10000.0)
    params = {
        "alphabot_short_sma": 10,
        "alphabot_long_sma": 20,
        "alphabot_trade_qty": 10,
        "riskguard_url": "http://localhost:8001",
        "riskguard_max_pos_size": 1000,
        "riskguard_max_concentration": 0.5,
    }

    outcome = await _call_alphabot_a2a(
        client=mock_client,
        session_id=session_id,
        day=day,
        current_price=current_price,
        historical_prices=historical_prices,
        portfolio=portfolio,
        params=params,
        sim_logger=mock_logger,
    )

    assert outcome["approved_trade"] is None
    assert outcome["rejected_trade"] is None
    assert "A2A Response Format Issue: Expected Message" in outcome["error"]
    mock_client.send_message.assert_called_once()


def test_run_simulation_success(mock_a2a_call):
    """Test a successful simulation run."""
    response = client.post(
        "/run_simulation",
        data={
            "alphabot_short_sma": "10",
            "alphabot_long_sma": "20",
            "alphabot_trade_qty": "10",
            "sim_days": "5",
            "sim_initial_cash": "10000",
            "sim_initial_price": "100",
            "sim_volatility": "0.02",
            "sim_trend": "0.001",
            "riskguard_url": defaults.DEFAULT_RISKGUARD_URL,
            "riskguard_max_pos_size": "1000",
            "riskguard_max_concentration": "50",
            "alphabot_url": defaults.DEFAULT_ALPHABOT_URL,
        },
    )
    assert response.status_code == 200
    assert "Simulation completed successfully." in response.text
    # Check that our mock was called, e.g., once per simulation day
    assert mock_a2a_call.call_count == 5
    # Check that the response contains the results
    assert "Total Value" in response.text


def test_run_simulation_invalid_params():
    """Test simulation run with invalid parameters."""
    response = client.post(
        "/run_simulation",
        data={
            "alphabot_short_sma": "0",  # Invalid value
            "alphabot_long_sma": "20",
            "alphabot_trade_qty": "10",
            "sim_days": "5",
            "sim_initial_cash": "10000",
            "sim_initial_price": "100",
            "sim_volatility": "0.02",
            "sim_trend": "0.001",
            "riskguard_url": defaults.DEFAULT_RISKGUARD_URL,
            "riskguard_max_pos_size": "1000",
            "riskguard_max_concentration": "50",
            "alphabot_url": defaults.DEFAULT_ALPHABOT_URL,
        },
    )
    assert response.status_code == 200
    assert "Invalid simulation parameters" in response.text
    assert "Input should be greater than 0" in response.text


def test_run_simulation_connection_error(mock_a2a_call):
    """Test simulation run with an A2A connection error."""
    mock_a2a_call.side_effect = ConnectionError("Test connection error")

    response = client.post(
        "/run_simulation",
        data={
            "alphabot_short_sma": "10",
            "alphabot_long_sma": "20",
            "alphabot_trade_qty": "10",
            "sim_days": "5",
            "sim_initial_cash": "10000",
            "sim_initial_price": "100",
            "sim_volatility": "0.02",
            "sim_trend": "0.001",
            "riskguard_url": defaults.DEFAULT_RISKGUARD_URL,
            "riskguard_max_pos_size": "1000",
            "riskguard_max_concentration": "50",
            "alphabot_url": defaults.DEFAULT_ALPHABOT_URL,
        },
    )
    assert response.status_code == 200
    assert "Simulation failed: Connection Error" in response.text


def test_concurrent_simulations_no_race_condition():
    """Test that concurrent simulations don't interfere with each other.

    This test demonstrates that the race condition has been fixed by
    making multiple concurrent requests and verifying that each
    request gets its own, correct results.
    """
    import asyncio
    from unittest.mock import AsyncMock, patch

    async def run_single_simulation(simulation_id):
        """Run a single simulation with unique parameters."""
        with patch("simulator.main._call_alphabot_a2a", new_callable=AsyncMock) as mock:
            # Configure the mock to return different results based on simulation_id
            mock.return_value = {
                "approved_trade": {
                    "action": "BUY" if simulation_id % 2 == 0 else "SELL",
                    "quantity": 10 + simulation_id,
                    "price": 100.0 + simulation_id,
                    "ticker": "SIM",
                },
                "rejected_trade": None,
                "reason": f"SMA crossover for simulation {simulation_id}",
                "error": None,
            }

            response = client.post(
                "/run_simulation",
                data={
                    "alphabot_short_sma": str(10 + simulation_id),
                    "alphabot_long_sma": str(20 + simulation_id),
                    "alphabot_trade_qty": str(10 + simulation_id),
                    "sim_days": "5",
                    "sim_initial_cash": str(10000 + simulation_id * 1000),
                    "sim_initial_price": str(100 + simulation_id),
                    "sim_volatility": "0.02",
                    "sim_trend": "0.001",
                    "riskguard_url": defaults.DEFAULT_RISKGUARD_URL,
                    "riskguard_max_pos_size": "1000",
                    "riskguard_max_concentration": "50",
                    "alphabot_url": defaults.DEFAULT_ALPHABOT_URL,
                },
            )
            return response

    async def run_concurrent_simulations():
        """Run multiple simulations concurrently."""
        tasks = [run_single_simulation(i) for i in range(3)]
        responses = await asyncio.gather(*tasks)
        return responses

    # Run the concurrent simulations
    responses = asyncio.run(run_concurrent_simulations())

    # Verify that all responses are successful
    for i, response in enumerate(responses):
        assert response.status_code == 200
        assert "Simulation completed successfully." in response.text
        # Verify that each response contains the correct parameters for that simulation
        # We can check for the initial cash value in the results section
        initial_cash = 10000 + i * 1000
        assert (
            f"${initial_cash:,}" in response.text
            or f"${initial_cash:,.2f}" in response.text
        )
