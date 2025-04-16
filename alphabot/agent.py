import json
import logging
from typing import AsyncGenerator, List

# ADK Imports
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.tools import BaseTool
from google.genai import types as genai_types

# Import defaults from the central config
from common.config import (
    DEFAULT_ALPHABOT_LONG_SMA,
    DEFAULT_ALPHABOT_SHORT_SMA,
    DEFAULT_ALPHABOT_TRADE_QTY,
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
    DEFAULT_RISKGUARD_URL,
    DEFAULT_TICKER,
)
from common.utils.indicators import calculate_sma

from .a2a_risk_tool import A2ARiskCheckTool

logger = logging.getLogger(__name__)


class AlphaBotAgent(BaseAgent):
    ticker: str
    _should_be_long: bool = False
    tools: List[BaseTool]

    def __init__(
        self,
        stock_ticker: str = DEFAULT_TICKER,
        agent_name: str = "AlphaBot",
        **kwargs,
    ):
        a2a_tool = A2ARiskCheckTool()
        agent_description = (
            "AlphaBot agent using SMA crossover strategy with A2A risk checks."
        )
        super().__init__(
            name=agent_name,
            description=agent_description,
            tools=[a2a_tool],
            ticker=stock_ticker,
            **kwargs,
        )
        self._should_be_long = False

    def _parse_and_validate_input(self, ctx: InvocationContext) -> dict | None:
        """Parses input JSON and performs basic validation."""
        logger.debug(f"[{self.name}] Parsing input context...")
        if (
            not ctx.user_content
            or not ctx.user_content.parts
            or not hasattr(ctx.user_content.parts[0], "text")
        ):
            logger.error(
                f"[{self.name}] ERROR - Input event text not found in ctx.user_content.parts[0].text."
            )
            return None

        input_text = ctx.user_content.parts[0].text
        try:
            input_data = json.loads(input_text)
            logger.debug(f"[{self.name}] Parsed input_data: {input_data}")
            return input_data
        except json.JSONDecodeError:
            logger.error(
                f"[{self.name}] ERROR - Failed to decode input JSON: {input_text[:100]}..."
            )
            return None

    def _extract_parameters(self, input_data: dict) -> dict:
        """Extracts strategy and risk parameters from input data, applying defaults."""
        logger.debug(f"[{self.name}] Extracting parameters...")
        params = {
            "historical_prices": input_data.get("historical_prices", []),
            "current_price": input_data.get("current_price", None),
            "portfolio_state": input_data.get("portfolio_state", {}),
            "short_sma_period": input_data.get(
                "short_sma_period", DEFAULT_ALPHABOT_SHORT_SMA
            ),
            "long_sma_period": input_data.get(
                "long_sma_period", DEFAULT_ALPHABOT_LONG_SMA
            ),
            "trade_quantity": input_data.get(
                "trade_quantity", DEFAULT_ALPHABOT_TRADE_QTY
            ),
            "riskguard_url": input_data.get("riskguard_url", DEFAULT_RISKGUARD_URL),
            "max_pos_size": input_data.get(
                "max_pos_size", DEFAULT_RISKGUARD_MAX_POS_SIZE
            ),
            "max_concentration": input_data.get(
                "max_concentration", DEFAULT_RISKGUARD_MAX_CONCENTRATION
            ),
        }
        logger.debug(
            f"[{self.name}] Using AlphaBot Params: short_sma={params['short_sma_period']}, "
            f"long_sma={params['long_sma_period']}, trade_qty={params['trade_quantity']}"
        )
        logger.debug(
            f"[{self.name}] Using RiskGuard Params: url={params['riskguard_url']}, "
            f"max_pos={params['max_pos_size']}, max_conc={params['max_concentration']}"
        )
        logger.debug(
            f"[{self.name}] Received historical_prices list with length: {len(params['historical_prices'])}"
        )
        return params

    def _calculate_indicators(
        self, historical_prices: List[float], short_period: int, long_period: int
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Calculates current and previous short/long SMAs."""
        logger.debug(f"[{self.name}] Calculating indicators...")
        sma_short = calculate_sma(historical_prices, short_period)
        sma_long = calculate_sma(historical_prices, long_period)

        prev_sma_short = None
        prev_sma_long = None
        if len(historical_prices) > max(short_period, long_period):
            previous_prices = historical_prices[:-1]
            prev_sma_short = calculate_sma(previous_prices, short_period)
            prev_sma_long = calculate_sma(previous_prices, long_period)

        if sma_short is not None and sma_long is not None:
            logger.debug(
                f"[{self.name}] Current SMA({short_period})={sma_short:.2f}, SMA({long_period})={sma_long:.2f}"
            )
        if prev_sma_short is not None and prev_sma_long is not None:
            logger.debug(
                f"[{self.name}] Previous SMA({short_period})={prev_sma_short:.2f}, SMA({long_period})={prev_sma_long:.2f}"
            )

        return sma_short, sma_long, prev_sma_short, prev_sma_long

    def _generate_signal(
        self,
        sma_short: float | None,
        sma_long: float | None,
        prev_sma_short: float | None,
        prev_sma_long: float | None,
    ) -> str | None:
        """Generates BUY/SELL signal based on SMA crossover."""
        logger.debug(f"[{self.name}] Generating signal...")
        if (
            sma_short is None
            or sma_long is None
            or prev_sma_short is None
            or prev_sma_long is None
        ):
            logger.info(f"[{self.name}] Not enough history for signal generation.")
            return None

        buy_cond1 = prev_sma_short <= prev_sma_long
        buy_cond2 = sma_short > sma_long
        logger.debug(
            f"[{self.name}] BUY Check: (Prev Short <= Prev Long) = {buy_cond1}, (Curr Short > Curr Long) = {buy_cond2}"
        )
        if buy_cond1 and buy_cond2:
            logger.info(f"[{self.name}] +++ BUY SIGNAL DETECTED +++")
            return "BUY"

        sell_cond1 = prev_sma_short >= prev_sma_long
        sell_cond2 = sma_short < sma_long
        logger.debug(
            f"[{self.name}] SELL Check: (Prev Short >= Prev Long) = {sell_cond1}, (Curr Short < Curr Long) = {sell_cond2}"
        )
        if sell_cond1 and sell_cond2:
            logger.info(f"[{self.name}] --- SELL SIGNAL DETECTED ---")
            return "SELL"

        logger.info(f"[{self.name}] No crossover signal conditions met.")
        return None

    def _determine_trade_proposal(
        self,
        signal: str | None,
        current_price: float,
        portfolio_state: dict,
        trade_quantity: int,
    ) -> dict | None:
        """Determines the trade proposal based on signal and current state."""
        logger.debug(
            f"[{self.name}] Determining trade proposal for signal: {signal}, current state (_should_be_long): {self._should_be_long}"
        )
        trade_proposal = None
        if signal == "BUY":
            if not self._should_be_long:
                trade_proposal = {
                    "action": "BUY",
                    "ticker": self.ticker,
                    "quantity": trade_quantity,
                    "price": current_price,
                }
                logger.info(
                    f"[{self.name}] Proposing {trade_proposal['action']} {trade_proposal['quantity']} {trade_proposal['ticker']} @ ${trade_proposal['price']:.2f} based on signal and not being long."
                )
            else:
                logger.info(
                    f"[{self.name}] BUY Signal ignored, already in desired long position."
                )
                logger.debug(
                    f"[{self.name}] _determine_trade_proposal: BUY signal ignored because _should_be_long is True."
                )
        elif signal == "SELL":
            if self._should_be_long:
                current_shares = portfolio_state.get("shares", 0)
                logger.debug(
                    f"[{self.name}] _determine_trade_proposal: Agent thinks it should be long. Current shares: {current_shares}"
                )
                if current_shares > 0:
                    trade_proposal = {
                        "action": "SELL",
                        "ticker": self.ticker,
                        "quantity": current_shares,
                        "price": current_price,
                    }
                    logger.info(
                        f"[{self.name}] Proposing SELL {trade_proposal['quantity']} {trade_proposal['ticker']} @ ${trade_proposal['price']:.2f} based on signal and being long."
                    )
                else:
                    self._should_be_long = False
                    logger.warning(
                        f"[{self.name}] SELL Signal, but no shares held. Correcting _should_be_long to False. No trade proposed."
                    )
            else:
                logger.info(
                    f"[{self.name}] SELL Signal ignored, already in desired short/flat position."
                )
                logger.debug(
                    f"[{self.name}] _determine_trade_proposal: SELL signal ignored because _should_be_long is False."
                )
        return trade_proposal

    async def _perform_risk_check(
        self,
        trade_proposal: dict,
        portfolio_state: dict,
        risk_params: dict,
        ctx: InvocationContext,
    ) -> dict | None:
        """Calls the A2A Risk Check tool and returns the result."""
        logger.info(f"[{self.name}] Performing A2A Risk Check...")

        tool_args = {
            "trade_proposal": trade_proposal,
            "portfolio_state": portfolio_state,
            "risk_params": risk_params,
        }
        logger.debug(f"[{self.name}] A2A Tool Args: {tool_args}")

        a2a_risk_tool_instance = next(
            (t for t in self.tools if isinstance(t, A2ARiskCheckTool)), None
        )

        if not a2a_risk_tool_instance:
            logger.error(f"[{self.name}] ERROR - A2A Risk check tool not found.")
            return None

        tool_event_generator = a2a_risk_tool_instance.run_async(
            args=tool_args, tool_context=ctx
        )

        risk_result = None
        async for tool_event in tool_event_generator:
            if tool_event.author == a2a_risk_tool_instance.name:
                response_parts = tool_event.get_function_responses()
                if response_parts:
                    risk_result = response_parts[0].response
                    logger.debug(f"[{self.name}] Extracted risk result: {risk_result}")
                    break
                else:
                    logger.warning(
                        f"[{self.name}] Warning: Event from {tool_event.author} did not contain FunctionResponse parts."
                    )
            else:
                logger.debug(
                    f"[{self.name}] Received intermediate event from author: {tool_event.author}"
                )

        if risk_result is None:
            logger.error(
                f"[{self.name}] Did not receive a valid result from A2A RiskGuard tool after iterating."
            )

        return risk_result

    def _process_risk_result(
        self, risk_result: dict, trade_proposal: dict, signal: str
    ) -> Event:
        """Processes the risk check result, updates state, and creates the final event."""
        logger.debug(
            f"[{self.name}] _process_risk_result: Received risk_result={risk_result}, Original signal='{signal}', Current _should_be_long={self._should_be_long}"
        )
        logger.debug(f"[{self.name}] Processing risk result: {risk_result}")
        approved = risk_result.get("approved", False)
        reason = risk_result.get("reason", "No reason provided.")
        logger.debug(
            f"[{self.name}] _process_risk_result: Risk check approved={approved}. Reason='{reason}'"
        )

        if approved:
            logger.debug(
                f"[{self.name}] _process_risk_result: Trade APPROVED. Updating _should_be_long based on signal '{signal}'."
            )
            logger.info(
                f"[{self.name}] Trade APPROVED by RiskGuard (via A2A). Reason: {reason}"
            )
            if signal == "BUY":
                self._should_be_long = True
            elif signal == "SELL":
                self._should_be_long = False
            logger.info(
                f"[{self.name}] State updated: _should_be_long = {self._should_be_long}"
            )
            logger.debug(
                f"[{self.name}] _process_risk_result: State updated to _should_be_long={self._should_be_long}"
            )
            final_event = Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[genai_types.Part(text=f"Trade Approved (A2A): {reason}")]
                ),
                actions=EventActions(state_delta={"approved_trade": trade_proposal}),
            )
        else:
            logger.info(
                f"[{self.name}] Trade REJECTED by RiskGuard (via A2A). Reason: {reason}"
            )
            logger.debug(
                f"[{self.name}] _process_risk_result: Trade REJECTED. State _should_be_long remains {self._should_be_long}."
            )
            logger.info(
                f"[{self.name}] State unchanged: _should_be_long = {self._should_be_long}"
            )
            final_event = Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[genai_types.Part(text=f"Trade Rejected (A2A): {reason}")]
                ),
                actions=EventActions(
                    state_delta={"rejected_trade_proposal": trade_proposal}
                ),
            )
        return final_event

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name} Agent] Running Check")

        input_data = self._parse_and_validate_input(ctx)
        if input_data is None:
            yield Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[genai_types.Part(text="Error: Invalid input data.")]
                ),
            )
            return

        params = self._extract_parameters(input_data)
        historical_prices = params["historical_prices"]
        current_price = params["current_price"]
        portfolio_state = params["portfolio_state"]
        short_sma_period = params["short_sma_period"]
        long_sma_period = params["long_sma_period"]
        trade_quantity = params["trade_quantity"]
        risk_params = {
            "riskguard_url": params["riskguard_url"],
            "max_pos_size": params["max_pos_size"],
            "max_concentration": params["max_concentration"],
        }

        if not historical_prices or current_price is None:
            logger.warning(
                f"[{self.name}] Insufficient market data after parameter extraction."
            )
            yield Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            text="No signal due to insufficient market data."
                        )
                    ]
                ),
            )
            return

        sma_short, sma_long, prev_sma_short, prev_sma_long = self._calculate_indicators(
            historical_prices, short_sma_period, long_sma_period
        )
        if sma_short is None or sma_long is None:
            logger.info(f"[{self.name}] Not enough history for SMAs.")
            yield Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[genai_types.Part(text="No signal yet (calculating SMAs).")]
                ),
            )
            return

        signal = self._generate_signal(
            sma_short, sma_long, prev_sma_short, prev_sma_long
        )
        if signal is None:
            reason = (
                "Not enough history"
                if (prev_sma_short is None or prev_sma_long is None)
                else "Conditions not met"
            )
            logger.info(
                f"[{self.name}] No signal generated this period. Reason: {reason}"
            )
            yield Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[genai_types.Part(text=f"No signal ({reason}).")]
                ),
            )
            return

        trade_proposal = self._determine_trade_proposal(
            signal, current_price, portfolio_state, trade_quantity
        )
        if trade_proposal is None:
            logger.info(
                f"[{self.name}] Signal ({signal}) generated, but no trade action needed based on current state."
            )
            yield Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[genai_types.Part(text="Signal generated, no action needed.")]
                ),
            )
            return

        # Yield event before calling the risk check tool
        yield Event(
            author=self.name,
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        text=f"Proposing {trade_proposal['action']} {trade_proposal['quantity']} {trade_proposal['ticker']} @ ${trade_proposal['price']:.2f} (via A2A Risk Check)"
                    )
                ]
            ),
        )

        risk_result = None
        try:
            risk_result = await self._perform_risk_check(
                trade_proposal, portfolio_state, risk_params, ctx
            )
        except Exception as e:
            logger.error(
                f"[{self.name}] Error during risk check tool call: {e}", exc_info=True
            )
            yield Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[genai_types.Part(text="Error during risk check.")]
                ),
            )
            return

        if risk_result is None:
            yield Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            text="Trade proposal failed (no A2A risk response)."
                        )
                    ]
                ),
            )
            return

        final_event = self._process_risk_result(risk_result, trade_proposal, signal)
        yield final_event


root_agent = AlphaBotAgent()
logger.info(
    "AlphaBot root_agent instance created (using real A2A tool). Parameters passed via metadata."
)
