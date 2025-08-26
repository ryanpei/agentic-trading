import unittest

from a2a.types import DataPart, Message, Part

from alphabot.agent_executor import AlphaBotAgentExecutor
from tests.test_agent_executor_base import AgentExecutorTestBase


class TestAlphaBotAgentExecutor(AgentExecutorTestBase):
    AGENT_EXECUTOR_CLASS = AlphaBotAgentExecutor
    AGENT_EXECUTOR_PATH = "alphabot.agent_executor"

    def test_execute_success_buy_decision(self):
        # Arrange
        self.mock_runner_instance.run_async.return_value = (
            self.adk_mock_alphabot_generator(
                final_state_delta={"approved_trade": {"action": "BUY", "quantity": 10}},
                final_reason="SMA crossover indicates buy signal.",
            )
        )
        request_message = Message(
            messageId="test_message_id",
            role="user",
            parts=[
                Part(
                    root=DataPart(
                        data={
                            "current_price": 155.0,
                            "historical_prices": [150.0, 151.0, 152.0],
                            "day": 1,
                        }
                    )
                ),
                Part(root=DataPart(data={"cash": 10000.0, "shares": 100})),
            ],
        )

        # Act
        self._run_execute(request_message, context_id="test_session_id")

        # Assert
        self.mock_task_updater_instance.start_work.assert_called_once()
        self.assertEqual(self.mock_runner_instance.run_async.call_count, 1)
        _, call_kwargs = self.mock_runner_instance.run_async.call_args
        self.assertEqual(call_kwargs["session_id"], "test_session_id")

        self.mock_task_updater_instance.add_artifact.assert_called_once()
        artifact_call_kwargs = (
            self.mock_task_updater_instance.add_artifact.call_args.kwargs
        )
        expected_artifact_data = {
            "approved": True,
            "trade_proposal": {"action": "BUY", "quantity": 10},
            "reason": "SMA crossover indicates buy signal.",
        }
        self.assertEqual(len(artifact_call_kwargs["parts"]), 1)
        self.assertEqual(
            artifact_call_kwargs["parts"][0].root.data, expected_artifact_data
        )
        self.mock_task_updater_instance.complete.assert_called_once()
        self.mock_task_updater_instance.failed.assert_not_called()

    def test_execute_missing_market_data(self):
        # Arrange
        self.mock_task_updater_instance.new_agent_message.return_value = Message(
            messageId="new_message",
            role="agent",
            parts=[
                Part(
                    root=DataPart(
                        data={
                            "error": "Invalid input: Missing market_data or portfolio_state"
                        }
                    )
                )
            ],
        )
        request_message = Message(
            messageId="test_message_id",
            role="user",
            parts=[Part(root=DataPart(data={"cash": 10000.0, "shares": 100}))],
        )

        # Act
        self._run_execute(request_message)

        # Assert
        self.mock_task_updater_instance.start_work.assert_called_once()
        self.mock_task_updater_instance.failed.assert_called_once()
        fail_kwargs = self.mock_task_updater_instance.failed.call_args.kwargs
        fail_message = fail_kwargs["message"]
        self.assertIn(
            "Invalid input: Missing market_data or portfolio_state",
            str(fail_message.parts[0].root.data),
        )
        self.mock_task_updater_instance.complete.assert_not_called()

    def test_execute_adk_runner_exception(self):
        # Arrange
        self.mock_runner_instance.run_async.side_effect = Exception("ADK Borked")
        self.mock_task_updater_instance.new_agent_message.return_value = Message(
            messageId="new_message",
            role="agent",
            parts=[Part(root=DataPart(data={"error": "ADK Agent error: ADK Borked"}))],
        )
        request_message = Message(
            messageId="test_message_id",
            role="user",
            parts=[
                Part(
                    root=DataPart(
                        data={
                            "current_price": 155.0,
                            "historical_prices": [150.0, 151.0, 152.0],
                        }
                    )
                ),
                Part(root=DataPart(data={"cash": 10000.0, "shares": 100})),
            ],
        )

        # Act
        self._run_execute(request_message)

        # Assert
        self.mock_task_updater_instance.start_work.assert_called_once()
        self.mock_task_updater_instance.failed.assert_called_once()
        fail_kwargs = self.mock_task_updater_instance.failed.call_args.kwargs
        fail_message = fail_kwargs["message"]
        self.assertIn(
            "ADK Agent error: ADK Borked", str(fail_message.parts[0].root.data)
        )
        self.mock_task_updater_instance.complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
