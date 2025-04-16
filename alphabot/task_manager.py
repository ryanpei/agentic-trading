# alphabot/task_manager.py
import json  # Keep json for dumping agent input
import logging
from typing import Any, AsyncIterable, Dict

from google.adk import Runner
from google.adk.agents import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from alphabot.a2a_risk_tool import A2ARiskCheckTool

# Assuming the main agent logic is in alphabot.agent.root_agent
# and the tool is in alphabot.a2a_risk_tool
from alphabot.agent import root_agent as alphabot_root_agent

# Import config defaults
from common.config import (
    DEFAULT_ALPHABOT_LONG_SMA,
    DEFAULT_ALPHABOT_SHORT_SMA,
    DEFAULT_ALPHABOT_TRADE_QTY,
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
    DEFAULT_RISKGUARD_URL,
)
from common.server import utils
from common.server.task_manager import InMemoryTaskManager
from common.types import (
    Artifact,
    DataPart,
    InvalidParamsError,
    JSONRPCResponse,
    Message,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    Task,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TextPart,
    UnsupportedOperationError,
)

logger = logging.getLogger(__name__)


class AlphaBotTaskManager(InMemoryTaskManager):
    """Manages tasks for the AlphaBot A2A server."""

    SUPPORTED_OUTPUT_MODES = [
        "text",
        "application/json",
        "data",
    ]  # Define supported output

    def __init__(self):
        super().__init__()
        # Initialize the ADK Runner for the Alphabot agent
        # The agent needs the risk tool, which reads the URL from env vars
        self._alphabot_agent = (
            alphabot_root_agent  # Assuming this is the Agent instance
        )
        self._runner = Runner(
            app_name="alphabot",
            agent=self._alphabot_agent,
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
            artifact_service=InMemoryArtifactService(),
        )
        logger.info("AlphaBotTaskManager initialized.")

    async def _update_store_and_notify(
        self, task_id: str, status: TaskStatus, artifacts: list[Artifact] | None = None
    ) -> Task:
        """Helper to update store and potentially send notifications."""
        # This overrides the base class method if needed, or call super().update_store
        task = await super().update_store(task_id, status, artifacts)
        # TODO: Add push notification logic here if implemented later
        return task

    async def on_send_task(
        self, request: SendTaskRequest
    ) -> SendTaskResponse | AsyncIterable[SendTaskResponse]:
        """Handles a non-streaming task request for AlphaBot."""
        task_params: TaskSendParams = request.params
        task_id = task_params.id
        session_id = task_params.sessionId

        # 1. Validate Input/Output Modes
        if not utils.are_modalities_compatible(
            task_params.acceptedOutputModes, self.SUPPORTED_OUTPUT_MODES
        ):
            logger.warning(
                f"Task {task_id}: Unsupported output modes requested: {task_params.acceptedOutputModes}"
            )
            return SendTaskResponse(
                id=request.id,
                error=utils.new_incompatible_types_error(request.id).error,
            )

        # 2. Upsert Task (sets initial state to SUBMITTED)
        await self.upsert_task(task_params)

        # 3. Extract Data from Request Parts
        market_data: Dict[str, Any] | None = None
        portfolio_state: Dict[str, Any] | None = None
        try:
            for part in task_params.message.parts:
                if isinstance(part, DataPart):
                    # Look for keys directly within the DataPart's data
                    if "market_data" in part.data:
                        market_data = part.data["market_data"]
                    elif "portfolio_state" in part.data:
                        portfolio_state = part.data["portfolio_state"]
                    # Add more specific checks if needed
            if market_data is None or portfolio_state is None:
                raise ValueError(
                    "Missing market_data or portfolio_state in request parts"
                )

            # Extract agent parameters from metadata (use imported defaults)
            agent_params = task_params.metadata or {}
            logger.debug(
                f"Task {task_id}: Received agent_params (metadata): {agent_params}"
            )  # Use debug
            short_sma = agent_params.get("short_sma", DEFAULT_ALPHABOT_SHORT_SMA)
            long_sma = agent_params.get("long_sma", DEFAULT_ALPHABOT_LONG_SMA)
            trade_qty = agent_params.get("trade_qty", DEFAULT_ALPHABOT_TRADE_QTY)
            # Also extract RiskGuard params to pass along to the agent/tool
            riskguard_url = agent_params.get("riskguard_url", DEFAULT_RISKGUARD_URL)
            max_pos_size = agent_params.get(
                "max_pos_size", DEFAULT_RISKGUARD_MAX_POS_SIZE
            )
            max_concentration = agent_params.get(
                "max_concentration", DEFAULT_RISKGUARD_MAX_CONCENTRATION
            )
            # Log parameters being used
            logger.debug(
                f"Task {task_id}: Using AlphaBot params - short_sma={short_sma}, long_sma={long_sma}, trade_qty={trade_qty}"
            )  # Use debug
            logger.debug(
                f"Task {task_id}: Using RiskGuard params - url={riskguard_url}, max_pos_size={max_pos_size}, max_concentration={max_concentration}"
            )  # Use debug

        except (ValueError, TypeError) as e:  # Add Pydantic ValidationError if using
            logger.error(
                f"Task {task_id}: Invalid input data or metadata - {e}", exc_info=True
            )  # Add exc_info
            await self._update_store_and_notify(
                task_id,
                TaskStatus(
                    state=TaskState.FAILED,
                    message=Message(
                        role="agent", parts=[TextPart(text=f"Invalid input: {e}")]
                    ),
                ),
            )
            # Return error response immediately
            return SendTaskResponse(
                id=request.id,
                error=InvalidParamsError(message=f"Invalid input data: {e}"),
            )

        # 4. Update Task Status to WORKING
        await self._update_store_and_notify(
            task_id,
            TaskStatus(
                state=TaskState.WORKING,
                message=Message(
                    role="agent", parts=[TextPart(text="Processing trade signal...")]
                ),
            ),
        )

        # 5. Prepare Input for ADK Agent
        agent_input_dict = {
            "historical_prices": market_data.get("historical_prices", []),
            "current_price": market_data.get("current_price"),
            "day": market_data.get("day"),
            "portfolio_state": portfolio_state,
            # Add extracted AlphaBot agent parameters
            "short_sma_period": short_sma,
            "long_sma_period": long_sma,
            "trade_quantity": trade_qty,
            # Add extracted RiskGuard parameters
            "riskguard_url": riskguard_url,
            "max_pos_size": max_pos_size,
            "max_concentration": max_concentration,
        }
        agent_input_dict = {k: v for k, v in agent_input_dict.items() if v is not None}
        logger.debug(
            f"Task {task_id}: Prepared agent_input_dict: {agent_input_dict}"
        )  # Use debug
        agent_input_json = json.dumps(agent_input_dict)
        agent_content = genai_types.Content(
            parts=[genai_types.Part(text=agent_input_json)]
        )

        # 6. Ensure ADK Session Exists
        logger.info(f"Task {task_id}: Checking/Creating ADK session: {session_id}")
        session = self._runner.session_service.get_session(
            app_name=self._runner.app_name,  # Use runner's app name
            user_id="a2a_user",  # Consistent user ID
            session_id=session_id,
        )
        if not session:
            logger.info(f"Task {task_id}: Creating NEW ADK session: {session_id}")
            session = self._runner.session_service.create_session(
                app_name=self._runner.app_name,
                user_id="a2a_user",
                session_id=session_id,
                state={},  # Start with empty state
            )
        else:
            logger.info(f"Task {task_id}: Using EXISTING ADK session: {session_id}")

        # 7. Run the ADK Agent and Process Results
        final_agent_event = None
        captured_state_delta = None
        final_result_dict: Dict[str, Any] = {}  # Initialize empty
        final_reason_text = "Reason not provided."  # Default reason

        try:
            logger.info(f"Task {task_id}: Running ADK Agent...")
            # Session is already guaranteed to exist here
            async for event in self._runner.run_async(
                user_id="a2a_user",  # Consistent user ID
                session_id=session_id,  # Use session ID from request
                new_message=agent_content,
            ):
                if event.actions and event.actions.state_delta:
                    captured_state_delta = event.actions.state_delta
                    logger.info(
                        f"Task {task_id}: Captured state_delta: {captured_state_delta}"
                    )
                if event.is_final_response():
                    final_agent_event = event
                    # Extract text reason if available in the final event
                    if event.content and event.content.parts:
                        text_part = next(
                            (
                                p
                                for p in event.content.parts
                                if hasattr(p, "text") and p.text
                            ),
                            None,
                        )
                        if text_part:
                            final_reason_text = text_part.text
                            logger.debug(
                                f"Task {task_id}: Captured reason text from final event: '{final_reason_text}'"
                            )  # Use debug

            # --- Determine Final Result ---
            # Priority 1: Use captured state_delta if it contains trade info
            if captured_state_delta:
                if "approved_trade" in captured_state_delta:
                    final_result_dict = {
                        "approved": True,
                        "trade_proposal": captured_state_delta["approved_trade"],
                        "reason": final_reason_text,  # Use captured reason
                    }
                    logger.info(
                        f"Task {task_id}: Using approved trade from state_delta with reason: '{final_reason_text}'."
                    )
                elif "rejected_trade_proposal" in captured_state_delta:
                    final_result_dict = {
                        "approved": False,
                        "trade_proposal": captured_state_delta[
                            "rejected_trade_proposal"
                        ],
                        "reason": final_reason_text,  # Use captured reason
                    }
                    logger.info(
                        f"Task {task_id}: Using rejected trade from state_delta with reason: '{final_reason_text}'."
                    )
                else:
                    logger.warning(
                        f"Task {task_id}: Captured state_delta did not contain expected trade keys: {captured_state_delta}"
                    )
                    # Fall through to check final event content if state_delta was not conclusive

            # Priority 2: If no trade info from state_delta, check final event content
            # This handles cases where the agent might just return text or a tool response directly
            if not final_result_dict and final_agent_event:
                if final_agent_event.content and final_agent_event.content.parts:
                    func_response_part = next(
                        (
                            p
                            for p in final_agent_event.content.parts
                            if p.function_response
                        ),
                        None,
                    )
                    # Check if text_part was already processed for reason
                    text_part_content = (
                        final_reason_text
                        if final_reason_text != "Reason not provided."
                        else None
                    )
                    if (
                        not text_part_content
                    ):  # If reason wasn't in final event, check again
                        text_part = next(
                            (
                                p
                                for p in final_agent_event.content.parts
                                if hasattr(p, "text") and p.text
                            ),
                            None,
                        )
                        if text_part:
                            text_part_content = text_part.text

                    if (
                        func_response_part
                        and func_response_part.function_response.name
                        == A2ARiskCheckTool.name
                    ):
                        # Tool response likely contains the final dict structure including approval/reason
                        # Use this as the final result ONLY if state_delta didn't provide trade info
                        final_result_dict = (
                            func_response_part.function_response.response
                        )
                        logger.info(
                            f"Task {task_id}: Using result directly from function response: {final_result_dict}"
                        )
                    elif text_part_content:
                        # Use text as a message, indicating no specific action/decision
                        final_result_dict = {
                            "status": "Info",
                            "message": text_part_content,
                        }
                        logger.info(
                            f"Task {task_id}: Using text from final event content as message: '{text_part_content}'"
                        )
                    else:
                        final_result_dict = {
                            "status": "Unknown",
                            "message": "Agent finished, unknown result format.",
                        }
                        logger.warning(
                            f"Task {task_id}: Final event content parts format not recognized: {final_agent_event.content.parts}"
                        )
                else:
                    final_result_dict = {
                        "status": "Error",
                        "details": "Agent final event had no content.",
                    }
                    logger.error(f"Task {task_id}: Final event had no content.")

            # Handle case where agent run finishes without a final event or usable state_delta
            if not final_result_dict:
                final_result_dict = {
                    "status": "Error",
                    "details": "Agent did not produce a usable final response or state delta.",
                }
                logger.error(
                    f"Task {task_id}: No usable final event or state delta captured."
                )

        except Exception as e:
            logger.exception(
                f"Task {task_id}: Error running AlphaBot agent"
            )  # Use logger.exception
            agent_error_details = f"Agent execution error: {e}"
            # Update task to FAILED immediately
            final_task = await self._update_store_and_notify(
                task_id,
                TaskStatus(
                    state=TaskState.FAILED,
                    message=Message(
                        role="agent",
                        parts=[
                            DataPart(
                                data={"status": "Error", "details": agent_error_details}
                            )
                        ],
                    ),
                ),
            )
            # Return error response immediately, including the failed task state
            return SendTaskResponse(id=request.id, result=final_task)

        # Determine final task state based on the outcome
        if final_result_dict.get("status") == "Error":
            final_task_state = TaskState.FAILED
            logger.error(
                f"Task {task_id}: Agent execution resulted in error state: {final_result_dict.get('details')}"
            )
        else:
            final_task_state = TaskState.COMPLETED
            logger.info(f"Task {task_id}: Agent execution completed.")

        # 8. Format Result and Update Task to Final State
        # Determine simple status message based on the result
        status_message_text = "Task completed."  # Default
        if final_result_dict.get("status") == "Error":
            status_message_text = (
                f"Task failed: {final_result_dict.get('details', 'Unknown error')}"
            )
        elif "approved" in final_result_dict:
            if final_result_dict["approved"]:
                status_message_text = "Trade approved."
            else:
                status_message_text = f"Trade rejected: {final_result_dict.get('reason', 'No reason provided.')}"
        elif "message" in final_result_dict:  # Handle info messages
            status_message_text = final_result_dict["message"]

        final_status = TaskStatus(
            state=final_task_state,
            # Use a simple TextPart for the status message
            message=Message(role="agent", parts=[TextPart(text=status_message_text)]),
        )
        # Create artifact containing the structured data
        final_artifact = Artifact(
            name="trade_decision", parts=[DataPart(data=final_result_dict)]
        )

        final_task = await self._update_store_and_notify(
            task_id, final_status, [final_artifact]  # Pass updated status and artifact
        )

        # 9. Return SendTaskResponse
        logger.info(f"Task {task_id}: Sending final response.")
        # Append history if requested (using base class helper)
        task_result_with_history = self.append_task_history(
            final_task, task_params.historyLength
        )
        return SendTaskResponse(id=request.id, result=task_result_with_history)

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        """Handles a streaming task request (currently not supported)."""
        logger.warning(
            f"Task {request.params.id}: Streaming not supported by AlphaBot."
        )
        # Immediately return an error response
        return JSONRPCResponse(
            id=request.id,
            error=UnsupportedOperationError(
                message="Streaming is not supported for this agent."
            ),
        )
