import json
import logging
from typing import Any, AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types as genai_types

from .rules import (
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
    RiskCheckResult,
    check_trade_risk_logic,
)

logger = logging.getLogger(__name__)


class RiskGuardAgent(BaseAgent):
    """
    ADK Agent implementing the RiskGuard logic.
    Designed to be hosted via an A2A server.
    """

    name: str = "RiskGuard"
    description: str = "Evaluates proposed trades against predefined risk rules."

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Processes a risk check request received via A2A (simulated via message content).
        """
        logger.info(f"Received risk check invocation")
        result_dict = {
            "approved": False,
            "reason": "Internal Error: Failed to process input.",
        }

        input_text = None
        if (
            ctx.user_content
            and ctx.user_content.parts
            and hasattr(ctx.user_content.parts[0], "text")
        ):
            input_text = ctx.user_content.parts[0].text

        if not input_text:
            logger.error("Input message text not found.")
            result_dict["reason"] = "Internal Error: Missing input message text."
        else:
            try:
                input_data = json.loads(input_text)
                trade_proposal = input_data.get("trade_proposal")
                portfolio_state = input_data.get("portfolio_state")
                max_pos_size = input_data.get(
                    "max_pos_size", DEFAULT_RISKGUARD_MAX_POS_SIZE
                )
                max_concentration = input_data.get(
                    "max_concentration", DEFAULT_RISKGUARD_MAX_CONCENTRATION
                )

                logger.info(
                    f"Parsed Input - Trade: {trade_proposal}, Portfolio: {portfolio_state}"
                )
                logger.info(
                    f"Using Risk Params: max_pos_size={max_pos_size}, max_concentration={max_concentration}"
                )

                if not trade_proposal or not portfolio_state:
                    logger.warning(
                        "Missing trade_proposal or portfolio_state in parsed input data."
                    )
                    result_dict["reason"] = (
                        "Internal Error: Missing input data in message."
                    )
                else:
                    result: RiskCheckResult = check_trade_risk_logic(
                        trade_proposal=trade_proposal,
                        portfolio_state=portfolio_state,
                        max_pos_size=max_pos_size,
                        max_concentration=max_concentration,
                    )
                    result_dict = {"approved": result.approved, "reason": result.reason}

            except json.JSONDecodeError:
                logger.error(
                    f"Failed to decode input JSON from message: {input_text[:100]}...",
                    exc_info=True,
                )
                result_dict["reason"] = (
                    "Internal Error: Invalid input data format in message."
                )
            except Exception as e:
                logger.error(f"Unexpected error processing input: {e}", exc_info=True)
                result_dict["reason"] = f"Internal Error: {e}"

        logger.info(f"Yielding result: {result_dict}")
        yield Event(
            author=self.name,
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="risk_check_result",
                            response=result_dict,
                        )
                    )
                ]
            ),
            turn_complete=True,
        )


root_agent = RiskGuardAgent()
