"""
FastAPI application for the Trading Simulator UI.

Provides a web interface to configure and run trading simulations using
AlphaBot (via A2A) for trade signals and RiskGuard (via A2A within AlphaBot)
for risk checks. Displays results including portfolio value, trades,
and performance metrics using Plotly charts.
"""

import asyncio
import json
import locale
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import plotly.graph_objects as go
from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

import common.config as defaults
from common.client import A2AClient
from common.config import DEFAULT_SIMULATOR_PORT
from common.types import (
    DataPart,
    Message,
    SendTaskResponse,
    Task,
    TaskSendParams,
    TaskState,
    TextPart,
)
from common.utils.indicators import calculate_sma

from .market import MarketDataSimulator
from .portfolio import PortfolioState, TradeAction

SIMULATOR_UI_LOGGER = "SimulatorUI"
SIMULATOR_LOGIC_LOGGER = "SimulatorLogic"
TRADE_DECISION_ARTIFACT_NAME = "trade_decision"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(SIMULATOR_UI_LOGGER)

try:
    import common.utils.indicators
    import simulator.market
    import simulator.portfolio
except ImportError as e:
    logger.error(f"Failed to import necessary application modules: {e}")
    logger.error(
        "Ensure you run uvicorn from the project root (agentic_trading) directory."
    )
    logger.error("Example: uvicorn simulator.main:app --reload --port 8003")
    sys.exit(1)

module_dir = Path(__file__).parent
templates_dir = module_dir / "templates"
static_dir = module_dir / "static"

simulation_status = {
    "is_running": False,
    "message": None,
    "is_error": False,
    "results": None,
    "params": {},
}


def format_currency(value: Optional[float]) -> str:
    """Formats an optional float value as currency, using locale settings if possible."""
    if value is None:
        return "N/A"
    try:
        return locale.currency(value, grouping=True)
    except ValueError:
        logger.debug("Locale formatting failed in format_currency, using fallback.")
        return f"${value:,.2f}"
    except Exception as e:
        logger.warning(
            f"Unexpected error during locale formatting: {e}. Using fallback."
        )
        return f"${value:,.2f}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events, including locale setting."""
    logger.info("Simulator UI starting up...")
    try:
        locale_setting = "en_US.UTF-8"
        locale.setlocale(locale.LC_ALL, locale_setting)
        logger.info(f"Default locale set to: {locale.getlocale(locale.LC_ALL)}")
    except locale.Error as e:
        logger.warning(
            f"Could not set default locale ('{locale_setting}') at startup: {e}. "
            "Check system locale settings. Using fallback currency formatting."
        )
    except Exception as e:
        logger.warning(
            f"Unexpected error setting locale ('{locale_setting}') at startup: {e}. Using fallback formatting."
        )
    yield
    logger.info("Simulator UI shutting down...")


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=str(templates_dir))
templates.env.filters["format_currency"] = format_currency
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _create_results_figure(
    results_df: pd.DataFrame, params: Dict[str, Any], trade_markers: Dict[str, List]
) -> go.Figure:
    """Creates the Plotly figure for simulation results."""
    MARKER_SIZE = 10
    MARKER_LINE_WIDTH = 1
    APPROVED_COLOR = "lime"
    REJECTED_COLOR = "red"
    MARKER_LINE_COLOR = "black"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["Price"],
            name="Price",
            line=dict(color="skyblue"),
            legendgroup="price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["SMA_Short"],
            name=f'SMA({params["alphabot_short_sma"]})',
            line=dict(color="orange", dash="dot"),
            legendgroup="price",
            legendrank=2,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["SMA_Long"],
            name=f'SMA({params["alphabot_long_sma"]})',
            line=dict(color="lightcoral", dash="dash"),
            legendgroup="price",
            legendrank=3,
        ),
        row=1,
        col=1,
    )

    if trade_markers["approved_buy_days"]:
        fig.add_trace(
            go.Scatter(
                x=trade_markers["approved_buy_days"],
                y=trade_markers["approved_buy_prices"],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    color=APPROVED_COLOR,
                    size=MARKER_SIZE,
                    line=dict(color=MARKER_LINE_COLOR, width=MARKER_LINE_WIDTH),
                ),
                name="Approved Buy",
                legendgroup="outcome",
                legendrank=4,
            ),
            row=1,
            col=1,
        )
    if trade_markers["rejected_buy_days"]:
        fig.add_trace(
            go.Scatter(
                x=trade_markers["rejected_buy_days"],
                y=trade_markers["rejected_buy_prices"],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    color=REJECTED_COLOR,
                    size=MARKER_SIZE,
                    line=dict(color=MARKER_LINE_COLOR, width=MARKER_LINE_WIDTH),
                ),
                name="Rejected Buy",
                legendgroup="outcome",
                legendrank=5,
            ),
            row=1,
            col=1,
        )
    if trade_markers["approved_sell_days"]:
        fig.add_trace(
            go.Scatter(
                x=trade_markers["approved_sell_days"],
                y=trade_markers["approved_sell_prices"],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    color=APPROVED_COLOR,
                    size=MARKER_SIZE,
                    line=dict(color=MARKER_LINE_COLOR, width=MARKER_LINE_WIDTH),
                ),
                name="Approved Sell",
                legendgroup="outcome",
                legendrank=6,
            ),
            row=1,
            col=1,
        )
    if trade_markers["rejected_sell_days"]:
        fig.add_trace(
            go.Scatter(
                x=trade_markers["rejected_sell_days"],
                y=trade_markers["rejected_sell_prices"],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    color=REJECTED_COLOR,
                    size=MARKER_SIZE,
                    line=dict(color=MARKER_LINE_COLOR, width=MARKER_LINE_WIDTH),
                ),
                name="Rejected Sell",
                legendgroup="outcome",
                legendrank=7,
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["TotalValue"],
            name="Total Value",
            line=dict(color="green"),
            legendgroup="portfolio",
            legendrank=8,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["Cash"],
            name="Cash",
            line=dict(color="lightgreen", dash="dash"),
            legendgroup="portfolio",
            legendrank=9,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["Shares"],
            name="Shares Held",
            line=dict(color="purple", dash="dot"),
            legendgroup="portfolio",
            legendrank=10,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            traceorder="grouped+reversed",
        ),
        autosize=True,
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(
        title_text="Shares", row=2, col=1, secondary_y=True, showgrid=False
    )

    return fig


class UILogHandler(logging.Handler):
    """Custom log handler to capture logs into a provided list for UI display."""

    def __init__(self, log_list: List[str]):
        super().__init__()
        self._log_list = log_list

    def emit(self, record):
        """Formats the record and appends it to the log list."""
        log_entry = self.format(record)
        self._log_list.append(log_entry)


async def _call_alphabot_a2a(
    client: A2AClient,
    session_id: str,
    day: int,
    current_price: float,
    historical_prices: List[float],
    portfolio: PortfolioState,
    params: Dict[str, Any],
    sim_logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Prepares and sends a task to the AlphaBot A2A server for a given simulation day.

    Args:
        client: The A2AClient instance for AlphaBot.
        session_id: The simulation session ID.
        day: The current simulation day.
        current_price: The current market price.
        historical_prices: List of historical prices.
        portfolio: The current PortfolioState object.
        params: Dictionary containing simulation and agent parameters.
        sim_logger: Logger instance for simulation logic.

    Returns:
        A dictionary containing the outcome:
        - 'approved_trade': Details of the approved trade (dict) or None.
        - 'rejected_trade': Details of the rejected trade (dict) or None.
        - 'reason': Reason text for approval/rejection (str) or None.
        - 'error': Error message if the A2A call failed (str) or None.
    """
    a2a_task_id = f"{session_id}-day{day}"
    market_data_part = DataPart(
        data={
            "market_data": {
                "day": day,
                "current_price": current_price,
                "historical_prices": historical_prices,
            }
        }
    )
    portfolio_state_part = DataPart(data={"portfolio_state": portfolio.__dict__})
    a2a_message = Message(role="user", parts=[market_data_part, portfolio_state_part])

    agent_params_metadata = {
        "short_sma": params["alphabot_short_sma"],
        "long_sma": params["alphabot_long_sma"],
        "trade_qty": params["alphabot_trade_qty"],
        "riskguard_url": params.get(
            "riskguard_url",
            os.environ.get("RISKGUARD_SERVICE_URL", defaults.DEFAULT_RISKGUARD_URL),
        ),
        "max_pos_size": params["riskguard_max_pos_size"],
        "max_concentration": params["riskguard_max_concentration"] / 100.0,
    }
    sim_logger.debug(f"  >> Metadata for A2A call: {agent_params_metadata}")

    a2a_params = TaskSendParams(
        id=a2a_task_id,
        sessionId=session_id,
        message=a2a_message,
        acceptedOutputModes=["data", "application/json"],
        metadata=agent_params_metadata,
    )

    sim_logger.info(f"--- Calling AlphaBot A2A Server (Task ID: {a2a_task_id}) ---")
    outcome = {
        "approved_trade": None,
        "rejected_trade": None,
        "reason": None,
        "error": None,
    }
    try:
        response: SendTaskResponse = await client.send_task(
            a2a_params.model_dump(exclude_none=True)
        )

        if response.error:
            sim_logger.error(
                f"A2A Error from AlphaBot: {response.error.code} - {response.error.message}"
            )
            outcome["error"] = f"A2A Error: {response.error.message}"
        elif response.result:
            task_result: Task = response.result
            sim_logger.info(
                f"A2A Task {task_result.id} completed with state: {task_result.status.state}"
            )

            result_data = None
            trade_decision_artifact = next(
                (
                    a
                    for a in task_result.artifacts
                    if a.name == TRADE_DECISION_ARTIFACT_NAME
                ),
                None,
            )

            if trade_decision_artifact and trade_decision_artifact.parts:
                art_part = trade_decision_artifact.parts[0]
                if isinstance(art_part, DataPart):
                    result_data = art_part.data
                    sim_logger.info(f"  >> Extracted Result Data: {result_data}")
                else:
                    sim_logger.warning(
                        f"  >> Unexpected part type in artifact: {type(art_part)}"
                    )
            else:
                sim_logger.warning("  >> 'trade_decision' artifact not found or empty.")
                if task_result.status.message and task_result.status.message.parts:
                    if isinstance(task_result.status.message.parts[0], TextPart):
                        sim_logger.info(
                            f"  >> Status Text: '{task_result.status.message.parts[0].text}'"
                        )

            if result_data and isinstance(result_data, dict):
                outcome["reason"] = result_data.get("reason", "Reason not provided.")
                if result_data.get("approved") is True:
                    outcome["approved_trade"] = result_data.get(
                        "trade_proposal", result_data
                    )
                    sim_logger.info(
                        f"    >>> Approved Trade: {outcome['approved_trade']} (Reason: {outcome['reason']})"
                    )
                elif result_data.get("approved") is False:
                    outcome["rejected_trade"] = result_data.get(
                        "trade_proposal", result_data
                    )
                    sim_logger.info(
                        f"    >>> Rejected Trade: {outcome['rejected_trade']} (Reason: {outcome['reason']})"
                    )
                else:
                    sim_logger.info(
                        f"  >> Received status update: {result_data.get('status', 'Unknown')}"
                    )
            else:
                sim_logger.warning(
                    "A2A response lacked expected result data structure."
                )
                outcome["error"] = "A2A Response Format Issue"
        else:
            sim_logger.error("A2A response had no result or error.")
            outcome["error"] = "Invalid A2A Response"

    except httpx.ConnectError as http_err:
        sim_logger.error(f"ConnectError to AlphaBot: {http_err}")
        outcome["error"] = f"AlphaBot Connection Failed: {http_err}"
        raise ConnectionError(f"AlphaBot Conn Fail: {http_err}") from http_err
    except httpx.RequestError as req_err:
        sim_logger.error(f"RequestError to AlphaBot: {req_err}")
        outcome["error"] = f"AlphaBot Request Failed: {req_err}"
        raise ConnectionError(f"AlphaBot Req Fail: {req_err}") from req_err
    except Exception as e:
        sim_logger.error(f"A2A Client/Processing Error: {e}", exc_info=True)
        outcome["error"] = f"A2A Processing Error: {e}"

    return outcome


async def run_simulation_async(params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the trading simulation and collects results. Returns dict with results or error."""
    logger.info(f"--- Starting Simulation with params: {params} ---")
    sim_log_list: List[str] = []
    ui_log_handler = UILogHandler(sim_log_list)
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    ui_log_handler.setFormatter(formatter)

    sim_logger = logging.getLogger(SIMULATOR_LOGIC_LOGGER)
    if not any(isinstance(h, UILogHandler) for h in sim_logger.handlers):
        sim_logger.addHandler(ui_log_handler)
    sim_logger.setLevel(logging.INFO)
    sim_logger.propagate = False

    daily_results = []
    signals = []
    approved_buy_days, approved_buy_prices = [], []
    rejected_buy_days, rejected_buy_prices = [], []
    approved_sell_days, approved_sell_prices = [], []
    rejected_sell_days, rejected_sell_prices = [], []

    try:
        sim_logger.info("Initializing simulation components...")
        portfolio = PortfolioState(cash=params["sim_initial_cash"])
        market_sim = MarketDataSimulator(
            initial_price=params["sim_initial_price"],
            volatility=params["sim_volatility"],
            trend=params["sim_trend"],
            history_size=params["alphabot_long_sma"] + 20,
        )

        alphabot_url = params.get(
            "alphabot_url",
            os.environ.get("ALPHABOT_SERVICE_URL", defaults.DEFAULT_ALPHABOT_URL),
        )
        sim_logger.info(f"Using AlphaBot Service URL: {alphabot_url}")
        sim_logger.info(f"Instantiating A2AClient for AlphaBot at: {alphabot_url}")
        try:
            client = A2AClient(url=alphabot_url)
        except Exception as e:
            sim_logger.error(f"Failed to initialize A2AClient for AlphaBot: {e}")
            raise ConnectionError(
                f"Could not connect or initialize A2A client for AlphaBot at {alphabot_url}"
            ) from e

        a2a_session_id = f"sim-session-{uuid.uuid4().hex[:8]}"
        sim_logger.info(f"Using A2A Session ID: {a2a_session_id}")

        initial_portfolio_str = f"Initial Portfolio: {portfolio}"
        sim_logger.info(initial_portfolio_str)
        signals.append({"day": 0, "log": initial_portfolio_str})

        total_days = params["sim_days"]
        sim_logger.info(f"Starting simulation loop for {total_days} days...")

        for day in range(1, total_days + 1):
            sim_logger.info(f"===== Day {day} =====")
            current_price = market_sim.next_price()
            historical_prices = market_sim.get_historical_prices()
            sim_logger.info(f"Market Data: Price = {format_currency(current_price)}")

            # Calculate SMAs
            sma_short = calculate_sma(historical_prices, params["alphabot_short_sma"])
            sma_long = calculate_sma(historical_prices, params["alphabot_long_sma"])

            portfolio.update_valuation(current_price)
            sim_logger.info(f"Portfolio (Start Day {day}): {portfolio}")

            # Store daily state BEFORE A2A call/trade
            daily_results.append(
                {
                    "Day": day,
                    "Price": current_price,
                    "SMA_Short": sma_short,
                    "SMA_Long": sma_long,
                    "Cash": portfolio.cash,
                    "Shares": portfolio.shares,
                    "HoldingsValue": portfolio.holdings_value,
                    "TotalValue": portfolio.total_value,
                }
            )

            a2a_outcome = await _call_alphabot_a2a(
                client=client,
                session_id=a2a_session_id,
                day=day,
                current_price=current_price,
                historical_prices=historical_prices,
                portfolio=portfolio,
                params=params,
                sim_logger=sim_logger,
            )

            approved_trade = a2a_outcome["approved_trade"]
            rejected_trade = a2a_outcome["rejected_trade"]
            reason_text = a2a_outcome["reason"]
            a2a_error = a2a_outcome["error"]

            trade_details = approved_trade or rejected_trade
            is_approved = approved_trade is not None
            action = trade_details.get("action") if trade_details else None

            signal_log_entry = {
                "day": day,
                "log": f"Price={format_currency(current_price)}",
            }

            if a2a_error:
                signal_log_entry["log"] += f" | A2A ERROR: {a2a_error}"
                sim_logger.error(f"A2A Error on day {day}: {a2a_error}")

            elif trade_details:
                qty = trade_details.get("quantity")
                price = trade_details.get("price")
                ticker = trade_details.get("ticker", "N/A")
                status = "Approved" if is_approved else "Rejected"
                reason = reason_text or (
                    "OK" if is_approved else "Reason not captured."
                )

                signal_log_entry[
                    "log"
                ] += f" | {action} {qty} {ticker} @ {format_currency(price)} | {status}: {reason}"

                if action == "BUY":
                    if is_approved:
                        approved_buy_days.append(day)
                        approved_buy_prices.append(current_price)
                    else:
                        rejected_buy_days.append(day)
                        rejected_buy_prices.append(current_price)
                elif action == "SELL":
                    if is_approved:
                        approved_sell_days.append(day)
                        approved_sell_prices.append(current_price)
                    else:
                        rejected_sell_days.append(day)
                        rejected_sell_prices.append(current_price)

                if is_approved:
                    sim_logger.info(
                        f"--- Executing Approved Trade: {action} {qty} @ {price} ---"
                    )
                    exec_action = trade_details.get("action")
                    exec_qty = trade_details.get("quantity")
                    exec_price = trade_details.get("price")

                    if exec_action and exec_qty is not None and exec_price is not None:
                        # Convert string action to TradeAction enum
                        trade_action_enum: Optional[TradeAction] = None
                        if exec_action.upper() == "BUY":
                            trade_action_enum = TradeAction.BUY
                        elif exec_action.upper() == "SELL":
                            trade_action_enum = TradeAction.SELL

                        if trade_action_enum:
                            trade_executed = portfolio.execute_trade(
                                action=trade_action_enum,
                                quantity=exec_qty,
                                price=exec_price,
                            )
                            if trade_executed:
                                portfolio.update_valuation(current_price)
                                sim_logger.info(
                                    f"Portfolio (Post-Trade Day {day}): {portfolio}"
                                )
                                signal_log_entry["log"] += " | Executed."
                            else:
                                sim_logger.warning(
                                    "--- Trade Execution FAILED (Insufficient funds/shares?) ---"
                                )
                                sim_logger.info(
                                    f"Portfolio (Post-Failed Trade Day {day}): {portfolio}"
                                )
                                signal_log_entry["log"] += " | Execution FAILED."
                        else:
                            # Log error if action string is not recognized
                            sim_logger.error(
                                f"--- Trade Execution SKIPPED - Unknown action '{exec_action}' received from AlphaBot ---"
                            )
                            signal_log_entry[
                                "log"
                            ] += (
                                f" | Execution SKIPPED (Unknown Action: {exec_action})."
                            )
                            sim_logger.info(
                                f"Portfolio (Skipped Trade Day {day}): {portfolio}"
                            )
                    else:
                        sim_logger.error(
                            f"--- Trade Execution SKIPPED - Missing details in approved_trade: {trade_details} ---"
                        )
                        signal_log_entry[
                            "log"
                        ] += " | Execution SKIPPED (Missing Data)."
                        sim_logger.info(
                            f"Portfolio (Skipped Trade Day {day}): {portfolio}"
                        )

                else:  # Trade was rejected
                    sim_logger.info(
                        f"--- Trade Rejected: {action} {qty} @ {price} (Reason: {reason}) ---"
                    )
                    sim_logger.info(
                        f"Portfolio (Rejected Trade Day {day}): {portfolio}"
                    )

            else:
                signal_log_entry["log"] += " | No trade proposed."
                sim_logger.info(f"Portfolio (No Trade Day {day}): {portfolio}")

            signals.append(signal_log_entry)

        sim_logger.info("--- Simulation End ---")
        sim_logger.info(f"Final Portfolio: {portfolio}")
        signals.append({"day": total_days + 1, "log": f"Final Portfolio: {portfolio}"})

        sim_logger.info("Preparing results dataframe and charts...")
        if not daily_results:
            results_df = pd.DataFrame(
                columns=[
                    "Day",
                    "Price",
                    "SMA_Short",
                    "SMA_Long",
                    "Cash",
                    "Shares",
                    "HoldingsValue",
                    "TotalValue",
                ]
            ).set_index("Day")
        else:
            results_df = pd.DataFrame(daily_results).set_index("Day")

        trade_markers = {
            "approved_buy_days": approved_buy_days,
            "approved_buy_prices": approved_buy_prices,
            "rejected_buy_days": rejected_buy_days,
            "rejected_buy_prices": rejected_buy_prices,
            "approved_sell_days": approved_sell_days,
            "approved_sell_prices": approved_sell_prices,
            "rejected_sell_days": rejected_sell_days,
            "rejected_sell_prices": rejected_sell_prices,
        }

        fig = _create_results_figure(results_df, params, trade_markers)
        charts = {"combined_chart_json": fig.to_json()}

        signals_log = "\n".join([f"Day {s['day']}: {s['log']}" for s in signals])
        detailed_log = "\n".join(sim_log_list)

        sim_logger.info("Simulation completed successfully.")
        return {
            "success": True,
            "final_portfolio": portfolio.__dict__,
            "charts": charts,
            "signals_log": signals_log,
            "detailed_log": detailed_log,
        }

    except ConnectionError as ce:
        error_msg = f"Connection Error: {ce}. Ensure AlphaBot A2A server is running at the specified URL."
        logger.error(error_msg)
        sim_logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "detailed_log": "\n".join(sim_log_list),
        }
    except Exception as e:
        error_msg = f"Simulation Error: {e}"
        logger.error(error_msg, exc_info=True)
        sim_logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "detailed_log": "\n".join(sim_log_list),
        }
    finally:
        if ui_log_handler in sim_logger.handlers:
            sim_logger.removeHandler(ui_log_handler)
        sim_logger.propagate = True


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    """Serves the main HTML page, passing simulation status and defaults."""
    logger.info("Serving root page.")
    template_context = {
        "request": request,
        "status": simulation_status,
        "params": simulation_status.get("params", {}),
        "DEFAULT_ALPHABOT_SHORT_SMA": defaults.DEFAULT_ALPHABOT_SHORT_SMA,
        "DEFAULT_ALPHABOT_LONG_SMA": defaults.DEFAULT_ALPHABOT_LONG_SMA,
        "DEFAULT_ALPHABOT_TRADE_QTY": defaults.DEFAULT_ALPHABOT_TRADE_QTY,
        "DEFAULT_ALPHABOT_URL": os.environ.get(
            "ALPHABOT_SERVICE_URL", defaults.DEFAULT_ALPHABOT_URL
        ),
        "DEFAULT_RISKGUARD_URL": os.environ.get(
            "RISKGUARD_SERVICE_URL", defaults.DEFAULT_RISKGUARD_URL
        ),
        "DEFAULT_RISKGUARD_MAX_POS_SIZE": defaults.DEFAULT_RISKGUARD_MAX_POS_SIZE,
        "DEFAULT_RISKGUARD_MAX_CONCENTRATION": defaults.DEFAULT_RISKGUARD_MAX_CONCENTRATION,
        "DEFAULT_SIM_DAYS": defaults.DEFAULT_SIM_DAYS,
        "DEFAULT_SIM_INITIAL_CASH": defaults.DEFAULT_SIM_INITIAL_CASH,
        "DEFAULT_SIM_INITIAL_PRICE": defaults.DEFAULT_SIM_INITIAL_PRICE,
        "DEFAULT_SIM_VOLATILITY": defaults.DEFAULT_SIM_VOLATILITY,
        "DEFAULT_SIM_TREND": defaults.DEFAULT_SIM_TREND,
    }
    return templates.TemplateResponse(
        request=request, name="index.html", context=template_context
    )


class SimulationRunParams(BaseModel):
    """Parameters for configuring and running a trading simulation, with validation."""

    alphabot_short_sma: int = Field(
        ..., gt=0, description="Short window for AlphaBot SMA (must be > 0)."
    )
    alphabot_long_sma: int = Field(
        defaults.DEFAULT_ALPHABOT_LONG_SMA,
        gt=0,
        description="Long window for AlphaBot SMA (must be > 0).",
    )
    alphabot_trade_qty: int = Field(
        defaults.DEFAULT_ALPHABOT_TRADE_QTY,
        gt=0,
        description="Quantity per trade for AlphaBot (must be > 0).",
    )
    sim_days: int = Field(
        defaults.DEFAULT_SIM_DAYS,
        gt=0,
        le=10000,
        description="Number of days to simulate (1-10000).",
    )
    sim_initial_cash: float = Field(
        defaults.DEFAULT_SIM_INITIAL_CASH,
        ge=0,
        description="Initial cash for simulation (must be >= 0).",
    )
    sim_initial_price: float = Field(
        defaults.DEFAULT_SIM_INITIAL_PRICE,
        gt=0,
        description="Initial price for market simulation (must be > 0).",
    )
    sim_volatility: float = Field(
        defaults.DEFAULT_SIM_VOLATILITY,
        ge=0,
        le=1.0,
        description="Volatility for market simulation (0.0-1.0).",
    )
    sim_trend: float = Field(
        defaults.DEFAULT_SIM_TREND,
        ge=-0.1,
        le=0.1,
        description="Trend for market simulation (-0.1 to 0.1).",
    )
    riskguard_url: str = Field(
        defaults.DEFAULT_RISKGUARD_URL, description="URL for RiskGuard service."
    )
    riskguard_max_pos_size: float = Field(
        defaults.DEFAULT_RISKGUARD_MAX_POS_SIZE,
        ge=0,
        description="Maximum position size allowed by RiskGuard (must be >= 0).",
    )
    riskguard_max_concentration: int = Field(
        int(defaults.DEFAULT_RISKGUARD_MAX_CONCENTRATION * 100),
        ge=0,
        le=100,
        description="Maximum portfolio concentration (%) allowed by RiskGuard (0-100).",
    )
    alphabot_url: str = Field(
        defaults.DEFAULT_ALPHABOT_URL, description="URL for AlphaBot service."
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


@app.post("/run_simulation")
async def handle_run_simulation(
    alphabot_short_sma: int = Form(...),
    alphabot_long_sma: int = Form(defaults.DEFAULT_ALPHABOT_LONG_SMA),
    alphabot_trade_qty: int = Form(defaults.DEFAULT_ALPHABOT_TRADE_QTY),
    sim_days: int = Form(defaults.DEFAULT_SIM_DAYS),
    sim_initial_cash: float = Form(defaults.DEFAULT_SIM_INITIAL_CASH),
    sim_initial_price: float = Form(defaults.DEFAULT_SIM_INITIAL_PRICE),
    sim_volatility: float = Form(defaults.DEFAULT_SIM_VOLATILITY),
    sim_trend: float = Form(defaults.DEFAULT_SIM_TREND),
    riskguard_url: str = Form(defaults.DEFAULT_RISKGUARD_URL),
    riskguard_max_pos_size: float = Form(defaults.DEFAULT_RISKGUARD_MAX_POS_SIZE),
    riskguard_max_concentration: int = Form(
        int(defaults.DEFAULT_RISKGUARD_MAX_CONCENTRATION * 100)
    ),
    alphabot_url: str = Form(defaults.DEFAULT_ALPHABOT_URL),
):
    """Handles the simulation run request, validating parameters via Pydantic."""
    if simulation_status["is_running"]:
        simulation_status["message"] = (
            "A simulation is already in progress. Please wait."
        )
        simulation_status["is_error"] = True
        return RedirectResponse("/", status_code=303)

    simulation_status["is_running"] = True
    simulation_status["message"] = "Simulation started..."
    simulation_status["is_error"] = False
    simulation_status["results"] = None

    try:
        sim_params = SimulationRunParams(
            alphabot_short_sma=alphabot_short_sma,
            alphabot_long_sma=alphabot_long_sma,
            alphabot_trade_qty=alphabot_trade_qty,
            sim_days=sim_days,
            sim_initial_cash=sim_initial_cash,
            sim_initial_price=sim_initial_price,
            sim_volatility=sim_volatility,
            sim_trend=sim_trend,
            riskguard_url=riskguard_url,
            riskguard_max_pos_size=riskguard_max_pos_size,
            riskguard_max_concentration=riskguard_max_concentration,
            alphabot_url=alphabot_url,
        )
        params_dict = sim_params.to_dict()
    except Exception as e:
        logger.error(f"Simulation parameter validation failed: {e}")
        simulation_status["message"] = f"Invalid simulation parameters: {e}"
        simulation_status["is_error"] = True
        simulation_status["is_running"] = False
        simulation_status["params"] = {
            k: v
            for k, v in locals().items()
            if k != "sim_params" and k != "params_dict" and k != "e"
        }
        return RedirectResponse("/", status_code=303)

    simulation_status["params"] = params_dict

    logger.info(f"Received simulation request with validated params: {params_dict}")
    logger.info(
        f"  >> handle_run_simulation: riskguard_max_pos_size = {params_dict['riskguard_max_pos_size']}"
    )

    results = await run_simulation_async(params_dict)

    if results.get("success"):
        simulation_status["message"] = "Simulation completed successfully."
        simulation_status["is_error"] = False
        simulation_status["results"] = results
    else:
        simulation_status["message"] = (
            f"Simulation failed: {results.get('error', 'Unknown error')}"
        )
        simulation_status["is_error"] = True
        simulation_status["results"] = {
            "detailed_log": results.get("detailed_log", "No detailed log available.")
        }

    simulation_status["is_running"] = False

    return RedirectResponse("/", status_code=303)


if __name__ == "__main__":
    import uvicorn

    logger.info("--- Starting FastAPI server for Simulator UI ---")
    logger.info("Ensure dependent A2A services are running:")
    logger.info(
        f"  RiskGuard: python -m agentic_trading.riskguard --port {defaults.DEFAULT_RISKGUARD_PORT}"
    )
    logger.info(
        f"  AlphaBot:  python -m agentic_trading.alphabot --port {defaults.DEFAULT_ALPHABOT_PORT}"
    )
    logger.info("Required Environment Variables (if not using defaults):")
    logger.info(f"  RISKGUARD_SERVICE_URL (default: {defaults.DEFAULT_RISKGUARD_URL})")
    logger.info(f"  ALPHABOT_SERVICE_URL (default: {defaults.DEFAULT_ALPHABOT_URL})")
    logger.info(f"--- Access UI at http://0.0.0.0:{DEFAULT_SIMULATOR_PORT} ---")

    uvicorn.run(
        "simulator.main:app", host="0.0.0.0", port=DEFAULT_SIMULATOR_PORT, reload=True
    )
