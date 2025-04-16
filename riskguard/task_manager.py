import json
import logging
from typing import Any, AsyncGenerator, Dict

from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types as genai_types

from common.config import (
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
)
from common.server.task_manager import InMemoryTaskManager
from common.types import (
    Artifact,
    DataPart,
    InternalError,
    InvalidRequestError,
    JSONRPCResponse,
    Message,
    SendTaskRequest,
    SendTaskResponse,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

from .agent import root_agent as riskguard_adk_agent

logger = logging.getLogger(__name__)


class RiskGuardTaskManager(InMemoryTaskManager):
    """Handles A2A task requests specifically for the RiskGuard ADK agent."""

    def __init__(self):
        super().__init__()
        self._adk_session_service = InMemorySessionService()
        self._adk_runner = Runner(
            app_name="RiskGuardInternal",
            agent=riskguard_adk_agent,
            session_service=self._adk_session_service,
        )
        logger.info("RiskGuardTaskManager initialized with ADK Runner.")

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        task_params = request.params
        task_id = task_params.id
        session_id = task_params.sessionId

        logger.info(f"Received task/send for Task ID: {task_id}")

        trade_proposal: Dict[str, Any] | None = None
        portfolio_state: Dict[str, Any] | None = None
        try:
            # 1. Extract trade proposal, portfolio state, and risk parameters from request
            if task_params.message and task_params.message.parts:
                for part in task_params.message.parts:
                    if isinstance(part, DataPart):
                        if "trade_proposal" in part.data:
                            trade_proposal = part.data["trade_proposal"]
                        if "portfolio_state" in part.data:
                            portfolio_state = part.data["portfolio_state"]
            if not trade_proposal or not portfolio_state:
                raise ValueError(
                    "Missing 'trade_proposal' or 'portfolio_state' in request parts"
                )

            risk_metadata = task_params.metadata or {}
            max_pos_size = risk_metadata.get(
                "max_pos_size", DEFAULT_RISKGUARD_MAX_POS_SIZE
            )
            max_concentration = risk_metadata.get(
                "max_concentration", DEFAULT_RISKGUARD_MAX_CONCENTRATION
            )
            logger.info(
                f"Task {task_id}: Using RiskGuard params from metadata - max_pos_size={max_pos_size}, max_concentration={max_concentration}"
            )

        except Exception as e:
            logger.error(
                f"Error extracting parameters for task {task_id}: {e}", exc_info=True
            )
            try:
                await self.upsert_task(task_params)
                await self.update_store(
                    task_id,
                    TaskStatus(
                        state=TaskState.FAILED,
                        message=Message(
                            role="agent", parts=[TextPart(text=f"Invalid input: {e}")]
                        ),
                    ),
                    None,
                )
            except Exception as update_err:
                logger.error(
                    f"Failed to update task store after parameter extraction error: {update_err}"
                )
            return SendTaskResponse(
                id=request.id,
                error=InvalidRequestError(message=f"Invalid parameters: {e}"),
            )

        # 2. Update A2A task status to WORKING
        await self.upsert_task(task_params)
        await self.update_store(
            task_id,
            TaskStatus(
                state=TaskState.WORKING,
                message=Message(
                    role="agent", parts=[TextPart(text="Checking risk...")]
                ),
            ),
            None,
        )

        # 3. Prepare input message for the ADK agent
        agent_input_data = {
            "trade_proposal": trade_proposal,
            "portfolio_state": portfolio_state,
            "max_pos_size": max_pos_size,
            "max_concentration": max_concentration,
        }
        agent_input_json = json.dumps(agent_input_data)
        initial_message = genai_types.Content(
            parts=[genai_types.Part(text=agent_input_json)]
        )
        logger.info(
            f"Task {task_id}: Sending data to ADK Agent via message: {agent_input_json[:200]}..."
        )

        # 4. Ensure ADK session exists (create if necessary)
        adk_session: Session | None = self._adk_session_service.get_session(
            app_name=self._adk_runner.app_name,
            user_id="a2a_user",
            session_id=session_id,
        )
        if adk_session is None:
            logger.info(f"Creating NEW ADK session: {session_id}")
            adk_session = self._adk_session_service.create_session(
                app_name=self._adk_runner.app_name,
                user_id="a2a_user",
                session_id=session_id,
                state={},
            )
        else:
            logger.info(f"Using EXISTING ADK session: {session_id}")

        final_adk_event: Event | None = None
        adk_result_json: str | None = None
        try:
            # 5. Run the ADK agent and process events to find the result
            logger.info(f"Running ADK Agent for Task ID: {task_id}")
            async for event in self._adk_runner.run_async(
                user_id="a2a_user",
                session_id=session_id,
                new_message=initial_message,
            ):

                if event.content and event.content.parts:
                    first_part = event.content.parts[0]

                    if (
                        hasattr(first_part, "function_response")
                        and first_part.function_response
                        and first_part.function_response.name == "risk_check_result"
                    ):
                        response_data = first_part.function_response.response
                        if isinstance(response_data, dict):
                            adk_result_json = json.dumps(response_data)
                            final_adk_event = event

                if event.is_final_response():
                    pass

            logger.info(f"ADK Agent finished for Task ID: {task_id}")
            if adk_result_json is None:
                logger.error(
                    f"Internal ADK agent did not produce a final text result for task {task_id}. Final event was: {final_adk_event}"
                )
                raise RuntimeError(
                    "Internal ADK agent did not produce a final text result."
                )

        except Exception as e:
            # 6. Handle errors during ADK agent execution
            logger.error(
                f"Error running internal ADK agent for task {task_id}: {e}",
                exc_info=True,
            )
            await self.update_store(
                task_id,
                TaskStatus(
                    state=TaskState.FAILED,
                    message=Message(
                        role="agent",
                        parts=[TextPart(text=f"Internal agent error: {e}")],
                    ),
                ),
                None,
            )
            return SendTaskResponse(
                id=request.id, error=InternalError(message=f"Internal agent error: {e}")
            )

        # 7. Format the successful result and update A2A task status to COMPLETED
        try:
            risk_result_dict = json.loads(adk_result_json)
        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse JSON result from internal agent for task {task_id}: {adk_result_json}"
            )
            risk_result_dict = {
                "approved": False,
                "reason": "Internal Error: Malformed result",
            }

        if risk_result_dict.get("approved", False):
            status_message_text = "Risk check approved."
        else:
            reason = risk_result_dict.get("reason", "No reason provided.")
            status_message_text = f"Risk check rejected: {reason}"

        final_task_status = TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(role="agent", parts=[TextPart(text=status_message_text)]),
        )
        final_artifact = Artifact(
            name="risk_assessment", parts=[DataPart(data=risk_result_dict)]
        )

        final_task = await self.update_store(
            task_id, final_task_status, [final_artifact]
        )

        logger.info(f"Sending COMPLETED Task response for ID: {task_id}")
        task_result_with_history = self.append_task_history(
            final_task, request.params.historyLength
        )
        return SendTaskResponse(id=request.id, result=task_result_with_history)

    async def on_send_task_subscribe(
        self, request
    ) -> AsyncGenerator[JSONRPCResponse, Any]:
        yield JSONRPCResponse(
            id=request.id,
            error=InternalError(message="Streaming not implemented for RiskGuard"),
        )

    async def on_get_task(self, request):
        task = await self.get_task(request.params.id)
        if task:
            return JSONRPCResponse(id=request.id, result=task)
        else:
            return JSONRPCResponse(
                id=request.id, error=InternalError(message="Task not found")
            )

    async def on_cancel_task(self, request):
        return JSONRPCResponse(
            id=request.id,
            error=InternalError(message="Cancel not implemented for RiskGuard"),
        )

    async def on_set_task_push_notification(self, request):
        return JSONRPCResponse(
            id=request.id,
            error=InternalError(
                message="Push notifications not implemented for RiskGuard"
            ),
        )

    async def on_get_task_push_notification(self, request):
        return JSONRPCResponse(
            id=request.id,
            error=InternalError(
                message="Push notifications not implemented for RiskGuard"
            ),
        )

    async def on_resubscribe_to_task(self, request):
        yield JSONRPCResponse(
            id=request.id,
            error=InternalError(message="Resubscribe not implemented for RiskGuard"),
        )
