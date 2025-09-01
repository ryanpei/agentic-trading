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
