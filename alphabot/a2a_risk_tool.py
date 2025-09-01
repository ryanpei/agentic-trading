import json
import logging
import os
import uuid
from typing import Any, AsyncGenerator, Dict

import httpx
from a2a.client import A2AClient, A2AClientHTTPError, A2AClientJSONError
from a2a.types import (
    DataPart,
    JSONRPCErrorResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TextPart,
)
from common.config import (
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
    DEFAULT_RISKGUARD_URL,
)
from google.adk.events import Event
from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types  # ADK uses google.genai.types
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class A2ARiskCheckTool(BaseTool):
    """
    ADK Tool that makes an A2A call to the RiskGuard service.
    """

    name: str = "a2a_risk_check"
    description: str = "Sends a trade proposal to the RiskGuard service for validation and returns the approval status and reason."
    risk_guard_url: str
    _httpx_client: httpx.AsyncClient

    def __init__(self, **kwargs):
        super().__init__(name=self.name, description=self.description, **kwargs)
        risk_guard_service_url = os.environ.get(
            "RISKGUARD_SERVICE_URL", DEFAULT_RISKGUARD_URL
        ).rstrip("/")
        self.risk_guard_url = risk_guard_service_url
        self._httpx_client = httpx.AsyncClient()
        logger.info(f"A2ARiskCheckTool initialized to target: {self.risk_guard_url}")

    async def close(self):
        """Close the httpx.AsyncClient when the tool is no longer needed."""
        if self._httpx_client:
            await self._httpx_client.aclose()
            logger.info("A2ARiskCheckTool httpx.AsyncClient closed.")

    def _get_declaration(self) -> genai_types.FunctionDeclaration:
        """
        Returns the ADK FunctionDeclaration for this tool.
        This describes the tool's interface to an LLM if it were to be called by one.
        """
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "trade_proposal": genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        description="Details of the trade being proposed, e.g., {'action': 'BUY', 'ticker': 'XYZ', 'quantity': 100, 'price': 50.0}",
                        properties={
                            "action": genai_types.Schema(
                                type=genai_types.Type.STRING, description="BUY or SELL"
                            ),
                            "ticker": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="Stock ticker symbol",
                            ),
                            "quantity": genai_types.Schema(
                                type=genai_types.Type.INTEGER,
                                description="Number of shares",
                            ),
                            "price": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Price per share",
                            ),
                        },
                        required=["action", "ticker", "quantity", "price"],
                    ),
                    "portfolio_state": genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        description="Current state of the portfolio, e.g., {'cash': 10000, 'shares': 50, 'total_value': 15000}",
                        properties={
                            "cash": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Current cash in portfolio",
                            ),
                            "shares": genai_types.Schema(
                                type=genai_types.Type.INTEGER,
                                description="Current shares held of the ticker",
                            ),
                            "total_value": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Total current value of the portfolio",
                            ),
                        },
                        required=["cash", "shares", "total_value"],
                    ),
                    "risk_params": genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        description="Risk parameters for the check, e.g., {'riskguard_url': 'http://...', 'max_pos_size': 10000, 'max_concentration': 0.5}",
                        properties={
                            "riskguard_url": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="URL of the RiskGuard service (optional, uses default if not provided).",
                            ),
                            "max_pos_size": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Maximum position size for a single trade.",
                            ),
                            "max_concentration": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Maximum portfolio concentration (0.0 to 1.0).",
                            ),
                        },
                        required=[],
                    ),
                },
                required=[
                    "trade_proposal",
                    "portfolio_state",
                ],
            ),
        )

    async def run_async(
        self, args: Dict[str, Any], tool_context: ToolContext
    ) -> AsyncGenerator[Event, None]:
        """Makes the actual A2A HTTP call."""
        invocation_id_short = tool_context.invocation_id[:8]
        logger.debug(
            f"[{self.name} Tool ({invocation_id_short})] Received args: {args}"
        )

        trade_proposal = args.get("trade_proposal")
        portfolio_state = args.get("portfolio_state")
        risk_params_from_args = args.get("risk_params", {})

        # Use risk_params from args if provided, otherwise fallback to AlphaBot's defaults
        risk_guard_target_url = risk_params_from_args.get(
            "riskguard_url", self.risk_guard_url
        )
        max_pos_size = risk_params_from_args.get(
            "max_pos_size", DEFAULT_RISKGUARD_MAX_POS_SIZE
        )
        max_concentration = risk_params_from_args.get(
            "max_concentration", DEFAULT_RISKGUARD_MAX_CONCENTRATION
        )

        logger.debug(
            f"[{self.name} Tool ({invocation_id_short})] Using Risk Params: url={risk_guard_target_url}, "
            f"max_pos_size={max_pos_size}, max_concentration={max_concentration}"
        )

        final_result_dict = {
            "approved": False,
            "reason": "A2A call failed or result not found.",
        }

        if not trade_proposal or not portfolio_state:
            logger.error(
                f"[{self.name} Tool ({invocation_id_short})] Error - Missing trade_proposal or portfolio_state in args."
            )
            final_result_dict["reason"] = "Tool Error: Missing input arguments."
            yield Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=self.name, response=final_result_dict
                            )
                        )
                    ]
                ),
                turn_complete=True,
            )
            return

        logger.info(
            f"[{self.name} Tool ({invocation_id_short})] Preparing A2A call to {risk_guard_target_url}"
        )

        # Construct the data payload for RiskGuard, including risk parameters
        risk_guard_payload = {
            "trade_proposal": trade_proposal,
            "portfolio_state": portfolio_state,
            "max_pos_size": max_pos_size,
            "max_concentration": max_concentration,
        }
        logger.debug(
            f"[{self.name} Tool ({invocation_id_short})] RiskGuard Payload: {risk_guard_payload}"
        )

        a2a_message = Message(
            context_id=tool_context._invocation_context.session.id,
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[
                Part(root=TextPart(text=json.dumps(risk_guard_payload))),
            ],
        )

        send_params = MessageSendParams(message=a2a_message)
        a2a_request = SendMessageRequest(id=str(uuid.uuid4()), params=send_params)

        a2a_sdk_client = A2AClient(
            httpx_client=self._httpx_client, url=risk_guard_target_url
        )
        try:
            response: SendMessageResponse = await a2a_sdk_client.send_message(
                a2a_request, http_kwargs={"timeout": 10}
            )
            logger.debug(
                f"[{self.name} Tool ({invocation_id_short})] Received A2A response: {response.model_dump_json(exclude_none=True)}"
            )
            root_response_part = response.root
            if isinstance(root_response_part, JSONRPCErrorResponse):
                actual_error = root_response_part.error
                logger.warning(
                    f"[{self.name} Tool ({invocation_id_short})] A2A call returned error: {actual_error.code} - {actual_error.message}"
                )
                final_result_dict["reason"] = (
                    f"A2A Error {actual_error.code}: {actual_error.message}"
                )
            elif isinstance(root_response_part, SendMessageSuccessResponse):
                # The result can be a Task or a Message
                response_result = root_response_part.result
                result_found = False

                if isinstance(response_result, Task):
                    task_result: Task = response_result
                    # First, check for artifacts (though RiskGuard doesn't send them in this setup)
                    if task_result.artifacts:
                        for artifact_item in task_result.artifacts:
                            if (
                                artifact_item.name == "risk_assessment"
                                and artifact_item.parts
                            ):
                                result_part_union = artifact_item.parts[0].root
                                if isinstance(result_part_union, DataPart):
                                    final_result_dict = result_part_union.data
                                    logger.info(
                                        f"[{self.name} Tool ({invocation_id_short})] Extracted 'risk_assessment' artifact from Task: {final_result_dict}"
                                    )
                                    result_found = True
                                    break

                    # If no artifact found, check task history or status message
                    if not result_found:
                        messages_to_check = []
                        if task_result.history:
                            messages_to_check.extend(task_result.history)
                        if task_result.status and task_result.status.message:
                            messages_to_check.append(task_result.status.message)

                        for msg in messages_to_check:
                            if msg.parts:
                                msg_part_root = msg.parts[0].root
                                result = self._extract_result_from_part(
                                    msg_part_root, invocation_id_short
                                )
                                if result:
                                    final_result_dict = result
                                    result_found = True
                                    break

                elif isinstance(response_result, Message):
                    # If RiskGuard sends a direct message (not wrapped in a Task)
                    if response_result.parts:
                        message_part_root = response_result.parts[0].root
                        result = self._extract_result_from_part(
                            message_part_root, invocation_id_short
                        )
                        if result:
                            final_result_dict = result
                            result_found = True

                if not result_found:
                    logger.warning(
                        f"[{self.name} Tool ({invocation_id_short})] 'risk_assessment' artifact or direct message result not found in A2A response."
                    )
                    final_result_dict["reason"] = (
                        "A2A Error: 'risk_assessment' artifact or direct message result not found."
                    )
                    final_result_dict["approved"] = False
            else:
                logger.error(
                    f"[{self.name} Tool ({invocation_id_short})] Unexpected A2A response structure. Root type: {type(root_response_part)}. Full response: {response.model_dump_json(exclude_none=True)}"
                )
                final_result_dict["reason"] = (
                    "A2A Error: Unexpected response structure."
                )

        except A2AClientHTTPError as e:
            logger.error(
                f"[{self.name} Tool ({invocation_id_short})] A2A SDK HTTP Error connecting to RiskGuard ({risk_guard_target_url}): {e.status_code} - {e.message}"
            )
            final_result_dict["reason"] = (
                f"A2A Network/HTTP Error: {e.status_code} - {e.message}. Is RiskGuard running?"
            )
        except A2AClientJSONError as e:
            logger.error(
                f"[{self.name} Tool ({invocation_id_short})] A2A SDK JSON Error from RiskGuard ({risk_guard_target_url}): {e.message}",
                exc_info=True,
            )
            final_result_dict["reason"] = f"A2A JSON Error: {e.message}"
        except ValidationError as e:
            logger.error(
                f"[{self.name} Tool ({invocation_id_short})] A2A Response Validation Error: {e}",
                exc_info=True,
            )
            final_result_dict["reason"] = (
                "A2A protocol error: Invalid response structure."
            )
        except Exception as e:
            logger.exception(
                f"[{self.name} Tool ({invocation_id_short})] Unexpected Error during A2A call"
            )
            final_result_dict["reason"] = f"A2A Client Error: {str(e)}"

        logger.info(
            f"[{self.name} Tool ({invocation_id_short})] Yielding final result: {final_result_dict}"
        )
        yield Event(
            author=self.name,
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=self.name,
                            response=final_result_dict,
                        )
                    )
                ]
            ),
            turn_complete=True,  # This tool completes its action in one go
        )

    def _extract_result_from_part(self, part_root, invocation_id_short: str) -> dict | None:
        """Extract result data from a message part."""
        if isinstance(part_root, TextPart) and part_root.text:
            try:
                parsed_data = json.loads(part_root.text)
                if (
                    isinstance(parsed_data, dict)
                    and "approved" in parsed_data
                    and "reason" in parsed_data
                ):
                    logger.info(
                        f"[{self.name} Tool ({invocation_id_short})] Extracted result from TextPart JSON: {parsed_data}"
                    )
                    return parsed_data
            except json.JSONDecodeError:
                logger.warning(
                    f"[{self.name} Tool ({invocation_id_short})] TextPart is not JSON: {part_root.text}"
                )
        elif isinstance(part_root, DataPart):
            logger.info(
                f"[{self.name} Tool ({invocation_id_short})] Extracted result from DataPart: {part_root.data}"
            )
            return part_root.data
        else:
            logger.warning(
                f"[{self.name} Tool ({invocation_id_short})] Part contains unsupported type: {type(part_root)}"
            )
        return None
