import logging
import os

import click
import common.config as defaults
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from common.utils.agent_utils import configure_a2a_server_params

# Import the specific AgentExecutor for AlphaBot
from .agent_executor import AlphaBotAgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", help="Host to bind the server to.")
@click.option("--port", type=int, help="Port to bind the server to.")
def main(host: str, port: int):
    """Runs the AlphaBot agent as an A2A server."""
    logger.info("Configuring AlphaBot A2A server...")
    host, port, a2a_base_url = configure_a2a_server_params(
        host_arg=host, port_arg=port, default_url=defaults.DEFAULT_ALPHABOT_URL
    )

    # Define the Agent Card for AlphaBot
    try:
        agent_card = AgentCard(
            name="AlphaBot Agent",
            description="Trading agent that analyzes market data and portfolio state to propose trades.",
            url=a2a_base_url,  # Use the publicly accessible URL
            version="1.0.0",
            capabilities=AgentCapabilities(
                streaming=False,  # AlphaBotTaskManager doesn't support streaming
                push_notifications=False,  # Not implemented
            ),
            skills=[
                AgentSkill(
                    id="provide_trade_signal",
                    name="Provide Trade Signal",
                    description="Analyzes market and portfolio data to decide whether to buy, sell, or hold.",
                    examples=[
                        "Given market data and portfolio, what trade should I make?"
                    ],
                    tags=[],
                )
            ],
            default_input_modes=["data"],
            default_output_modes=["data"],
        )
    except Exception:
        logger.exception("Error creating AgentCard")
        raise

    # Instantiate the AlphaBot AgentExecutor
    try:
        agent_executor = AlphaBotAgentExecutor()
    except Exception:
        logger.exception("Error initializing AlphaBotAgentExecutor")
        raise

    # Instantiate the A2AStarletteApplication
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=task_store
    )
    try:
        app_builder = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )
    except Exception:
        logger.exception("Error initializing A2AStarletteApplication")
        raise

    # Start the Server
    import uvicorn

    logger.info(f"Starting AlphaBot A2A server on http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server.")
    uvicorn.run(app_builder.build(), host=host, port=port)


if __name__ == "__main__":
    # Example: python -m alphabot --port 8081
    main()
