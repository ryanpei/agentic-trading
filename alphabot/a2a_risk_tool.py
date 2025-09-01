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
    Role,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
)
from common.config import (
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
    DEFAULT_RISKGUARD_URL,
)
from common.models import TradeProposal, PortfolioState, RiskCheckPayload
from common.utils.agent_utils import create_a2a_message_from_payload
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

    async def run_async(self, **kwargs: Any) -> AsyncGenerator[Event, None]:
        """Makes the actual A2A HTTP call."""
        tool_context: ToolContext = kwargs["tool_context"]
        args: Dict[str, Any] = kwargs["args"]
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

        # --- REFACTORED SECTION ---
        # 1. Create Pydantic models from your data
        trade_proposal_model = TradeProposal(**trade_proposal)
        portfolio_state_model = PortfolioState(**portfolio_state)

        risk_payload = RiskCheckPayload(
            trade_proposal=trade_proposal_model,
            portfolio_state=portfolio_state_model,
            max_pos_size=max_pos_size,
            max_concentration=max_concentration,
        )

        # 2. Use the new helper to create the A2A Message
        a2a_message = create_a2a_message_from_payload(
            payload=risk_payload,
            role=Role.user,
            context_id=tool_context._invocation_context.session.id,
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
                response_result = root_response_part.result
                if isinstance(response_result, Message):
                    if response_result.parts and isinstance(
                        response_result.parts[0].root, DataPart
                    ):
                        final_result_dict = response_result.parts[0].root.data
                    else:
                        final_result_dict["reason"] = (
                            "RiskGuard returned an invalid message format."
                        )
                else:
                    raise TypeError(
                        f"Expected a Message from RiskGuard, but got {type(response_result)}"
                    )
            else:
                logger.error(
                    f"[{self.name} Tool ({invocation_id_short})] Unexpected A2A response structure. Root type: {type(root_response_part)}."
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
