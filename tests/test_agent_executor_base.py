import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message


class AgentExecutorTestBase(unittest.TestCase):
    """Base class for Agent Executor tests to reduce code duplication."""

    AGENT_EXECUTOR_CLASS = None  # Must be overridden by subclasses
    AGENT_EXECUTOR_PATH = ""  # e.g., "alphabot.agent_executor"

    def setUp(self):
        self.maxDiff = None
        # Patch the logger to avoid log output during tests
        patcher = patch(f"{self.AGENT_EXECUTOR_PATH}.logger", new=MagicMock())
        self.addCleanup(patcher.stop)
        patcher.start()

        # Centralized mock setup
        self.mock_runner_patcher = patch(f"{self.AGENT_EXECUTOR_PATH}.Runner")
        self.mock_task_updater_patcher = patch(
            f"{self.AGENT_EXECUTOR_PATH}.TaskUpdater"
        )

        self.mock_runner = self.mock_runner_patcher.start()
        self.mock_task_updater = self.mock_task_updater_patcher.start()

        self.mock_runner_instance = self.mock_runner.return_value
        self.mock_task_updater_instance = self.mock_task_updater.return_value

        # Common mock behaviors
        self.mock_runner_instance.session_service.get_session = AsyncMock(
            return_value=None
        )
        self.mock_runner_instance.session_service.create_session = AsyncMock(
            return_value=MagicMock(id="test_session_id")
        )
        self.mock_task_updater_instance.add_artifact = AsyncMock()
        self.mock_task_updater_instance.complete = AsyncMock()
        self.mock_task_updater_instance.failed = AsyncMock()
        self.mock_task_updater_instance.submit = AsyncMock()
        self.mock_task_updater_instance.start_work = AsyncMock()
        self.mock_task_updater_instance.new_agent_message.return_value = Message(
            messageId="new_message", role="agent", parts=[]
        )

    def tearDown(self):
        self.mock_runner_patcher.stop()
        self.mock_task_updater_patcher.stop()

    def _run_execute(
        self, request_message, task_id="test_task", context_id="test_context"
    ):
        """Helper to run the executor's execute method."""
        executor = self.AGENT_EXECUTOR_CLASS()
        executor._adk_runner = self.mock_runner_instance  # Override with mock

        request_obj = MagicMock()
        request_obj.message = request_message
        context = RequestContext(
            task_id=task_id,
            context_id=context_id,
            request=request_obj,
        )
        event_queue = EventQueue()
        asyncio.run(executor.execute(context, event_queue))

    @staticmethod
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

    @staticmethod
    async def adk_mock_riskguard_generator(result_name, result_data):
        """Mocks the async generator for RiskGuard."""
        function_response_part = MagicMock()
        function_response_part.function_response.name = result_name
        function_response_part.function_response.response = result_data
        yield MagicMock(
            content=MagicMock(parts=[function_response_part]),
            is_final_response=lambda: True,
        )
