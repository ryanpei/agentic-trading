# alphabot/a2a_risk_tool.py

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict

import httpx
from google.adk.events import Event
from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types
from pydantic import ValidationError

from common.config import (
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
    DEFAULT_RISKGUARD_URL,
)
from common.types import DataPart, JSONRPCResponse, Message, Task, TaskSendParams

logger = logging.getLogger(__name__)


class A2ARiskCheckTool(BaseTool):
    """
    ADK Tool that makes an A2A call to the RiskGuard service.
    Relies on RiskGuard returning results via a 'risk_assessment' artifact.
    """

    name: str = "a2a_risk_check"
    description: str = "Sends a trade proposal to the RiskGuard service for validation."
    risk_guard_url: str

    def __init__(self, **kwargs):
        super().__init__(name=self.name, description=self.description, **kwargs)
        risk_guard_service_url = os.environ.get(
            "RISKGUARD_SERVICE_URL", DEFAULT_RISKGUARD_URL
        )
        self.risk_guard_url = risk_guard_service_url.rstrip("/") + "/"
        logger.info(f"A2ARiskCheckTool initialized to target: {self.risk_guard_url}")

    async def run_async(
        self, args: Dict[str, Any], tool_context: ToolContext
    ) -> AsyncGenerator[Event, None]:
        """Makes the actual A2A HTTP call."""
        logger.debug(f"[{self.name} Tool] Received args: {args}")
        trade_proposal = args.get("trade_proposal")
        portfolio_state = args.get("portfolio_state")
        risk_params = args.get("risk_params", {})
        logger.debug(f"[{self.name} Tool] Extracted risk_params: {risk_params}")

        risk_guard_target_url = risk_params.get("riskguard_url", self.risk_guard_url)
        max_pos_size = risk_params.get("max_pos_size", DEFAULT_RISKGUARD_MAX_POS_SIZE)
        max_concentration = risk_params.get(
            "max_concentration", DEFAULT_RISKGUARD_MAX_CONCENTRATION
        )
        logger.debug(
            f"[{self.name} Tool] Using Risk Params: url={risk_guard_target_url}, "
            f"max_pos_size={max_pos_size}, max_concentration={max_concentration}"
        )

        # Default error result dictionary
        final_result_dict = {
            "approved": False,
            "reason": "A2A call failed or result not found.",
        }

        if not trade_proposal or not portfolio_state:
            logger.error(
                f"[{self.name} Tool] Error - Missing trade_proposal or portfolio_state in args."
            )
            final_result_dict["reason"] = "Tool Error: Missing input arguments."
            # Yield error event and return immediately
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
                turn_complete=True,  # Indicate this tool step failed
            )
            return

        logger.info(f"[{self.name} Tool] Preparing A2A call to {risk_guard_target_url}")

        risk_metadata = {
            "max_pos_size": max_pos_size,
            "max_concentration": max_concentration,
        }
        logger.debug(f"[{self.name} Tool] Risk metadata: {risk_metadata}")

        a2a_params = TaskSendParams(
            id=f"riskcheck-{tool_context.invocation_id[:8]}",
            sessionId=tool_context.session.id,
            message=Message(
                role="user",
                parts=[
                    DataPart(data={"trade_proposal": trade_proposal}),
                    DataPart(data={"portfolio_state": portfolio_state}),
                ],
            ),
            metadata=risk_metadata,
        )
        a2a_request_payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": a2a_params.model_dump(exclude_none=True),
            "id": tool_context.invocation_id,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    risk_guard_target_url, json=a2a_request_payload
                )
                response.raise_for_status()
                a2a_response_json = response.json()
                logger.debug(
                    f"[{self.name} Tool] Received A2A response: {a2a_response_json}"
                )

                a2a_response = JSONRPCResponse(**a2a_response_json)

                if a2a_response.error:
                    logger.warning(
                        f"[{self.name} Tool] A2A call returned error: {a2a_response.error}"
                    )
                    final_result_dict["reason"] = (
                        f"A2A Error {a2a_response.error.code}: {a2a_response.error.message}"
                    )
                elif a2a_response.result:
                    task_result = Task(**a2a_response.result)
                    result_found = False  # Flag to track if we found the result

                    if task_result.artifacts:
                        for artifact in task_result.artifacts:
                            # Check for the specific artifact name and ensure it has parts
                            if artifact.name == "risk_assessment" and artifact.parts:
                                # Assume the first part is the DataPart we need
                                result_part = artifact.parts[0]
                                if isinstance(result_part, DataPart):
                                    final_result_dict = result_part.data
                                    logger.debug(
                                        f"[{self.name} Tool] Extracted result from Artifact: {final_result_dict}"
                                    )
                                    result_found = True
                                    break  # Exit loop once result is found
                                else:
                                    logger.warning(
                                        f"[{self.name} Tool] Found 'risk_assessment' artifact, "
                                        f"but first part is not DataPart (type: {type(result_part)})."
                                    )
                        # If loop finished without finding the specific artifact/part
                        if not result_found:
                            logger.warning(
                                f"[{self.name} Tool] No 'risk_assessment' DataPart artifact found in response."
                            )
                            final_result_dict["reason"] = (
                                "A2A Error: RiskGuard result artifact not found or invalid."
                            )
                    else:
                        logger.warning(
                            f"[{self.name} Tool] A2A response task contained no artifacts."
                        )
                        final_result_dict["reason"] = (
                            "A2A Error: RiskGuard response task had no artifacts."
                        )

                else:  # Handles case where response has neither 'error' nor 'result'
                    logger.error(
                        f"[{self.name} Tool] Invalid JSON-RPC response (no result or error): {a2a_response_json}"
                    )
                    final_result_dict["reason"] = (
                        "A2A Error: Invalid JSON-RPC response format."
                    )

        except httpx.RequestError as e:
            logger.error(
                f"[{self.name} Tool] HTTP Request Error connecting to RiskGuard ({risk_guard_target_url}): {e}"
            )
            final_result_dict["reason"] = (
                f"A2A Network Error: Could not connect to {risk_guard_target_url}. Is RiskGuard running?"
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                f"[{self.name} Tool] HTTP Status Error from RiskGuard ({risk_guard_target_url}): {e.response.status_code}",
                exc_info=True,
            )
            try:
                error_detail = e.response.json()
            except json.JSONDecodeError:
                error_detail = e.response.text
            final_result_dict["reason"] = (
                f"A2A HTTP Error {e.response.status_code}: {error_detail}"
            )
        except ValidationError as e:
            logger.error(
                f"[{self.name} Tool] A2A Response Validation Error: {e}", exc_info=True
            )
            final_result_dict["reason"] = (
                "A2A protocol error: Invalid response structure."
            )
        except Exception as e:
            logger.exception(f"[{self.name} Tool] Unexpected Error during A2A call")
            final_result_dict["reason"] = f"A2A Client Error: {e}"

        # Yield the final result back to AlphaBot
        logger.info(f"[{self.name} Tool] Yielding final result: {final_result_dict}")
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
        )
