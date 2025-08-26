import unittest

from a2a.types import DataPart, Message, Part

from riskguard.agent_executor import RiskGuardAgentExecutor
from tests.test_agent_executor_base import AgentExecutorTestBase


class TestRiskGuardAgentExecutor(AgentExecutorTestBase):
    AGENT_EXECUTOR_CLASS = RiskGuardAgentExecutor
    AGENT_EXECUTOR_PATH = "riskguard.agent_executor"

    def test_execute_success_approved(self):
        # Arrange
        self.mock_runner_instance.run_async.return_value = (
            self.adk_mock_riskguard_generator(
                result_name="risk_check_result",
                result_data={"approved": True, "reason": "Within risk parameters."},
            )
        )
        request_message = Message(
            messageId="test_message_id",
            role="user",
            parts=[
                Part(
                    root=DataPart(
                        data={
                            "trade_proposal": {"action": "BUY", "quantity": 10},
                            "portfolio_state": {"cash": 10000.0, "shares": 100},
                        }
                    )
                )
            ],
        )

        # Act
        self._run_execute(request_message, context_id="test_session_id")

        # Assert
        self.mock_task_updater_instance.start_work.assert_called_once()
        self.assertEqual(self.mock_runner_instance.run_async.call_count, 1)
        self.mock_task_updater_instance.add_artifact.assert_called_once()
        artifact_call_kwargs = (
            self.mock_task_updater_instance.add_artifact.call_args.kwargs
        )
        expected_artifact = {"approved": True, "reason": "Within risk parameters."}
        self.assertEqual(artifact_call_kwargs["parts"][0].root.data, expected_artifact)
        self.mock_task_updater_instance.complete.assert_called_once()
        self.mock_task_updater_instance.failed.assert_not_called()

    def test_execute_missing_trade_proposal(self):
        # Arrange
        self.mock_task_updater_instance.new_agent_message.return_value = Message(
            messageId="new_message",
            role="agent",
            parts=[Part(root=DataPart(data={"error": "Invalid input data"}))],
        )
        request_message = Message(
            messageId="test_message_id",
            role="user",
            parts=[
                Part(
                    root=DataPart(
                        data={"portfolio_state": {"cash": 10000.0, "shares": 100}}
                    )
                )
            ],
        )

        # Act
        self._run_execute(request_message)

        # Assert
        self.mock_task_updater_instance.start_work.assert_called_once()
        self.mock_task_updater_instance.failed.assert_called_once()
        fail_kwargs = self.mock_task_updater_instance.failed.call_args.kwargs
        fail_message = fail_kwargs["message"]
        self.assertIn("Invalid input data", str(fail_message.parts[0].root.data))
        self.mock_runner_instance.run_async.assert_not_called()

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
                            "trade_proposal": {"action": "BUY", "quantity": 10},
                            "portfolio_state": {"cash": 10000.0, "shares": 100},
                        }
                    )
                )
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
