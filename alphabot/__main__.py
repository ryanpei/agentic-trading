import logging

import click

import common.config as defaults
from common.server import A2AServer
from common.types import AgentCapabilities, AgentCard, AgentSkill, MissingAPIKeyError

# Import the specific Task Manager for AlphaBot
from .task_manager import AlphaBotTaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the server to.")
@click.option(
    "--port", default=defaults.DEFAULT_ALPHABOT_PORT, help="Port to bind the server to."
)
def main(host, port):
    """Runs the AlphaBot agent as an A2A server."""
    logger.info("Configuring AlphaBot A2A server...")

    # Define the Agent Card for AlphaBot
    try:
        agent_card = AgentCard(
            name="AlphaBot Agent",
            description="Trading agent that analyzes market data and portfolio state to propose trades.",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            capabilities=AgentCapabilities(
                streaming=False,  # AlphaBotTaskManager doesn't support streaming
                pushNotifications=False,  # Not implemented
            ),
            skills=[
                AgentSkill(
                    id="provide_trade_signal",
                    name="Provide Trade Signal",
                    description="Analyzes market and portfolio data to decide whether to buy, sell, or hold.",
                    examples=[
                        "Given market data and portfolio, what trade should I make?"
                    ],
                )
            ],
            defaultInputModes=[
                "data"
            ],  # Expects market/portfolio state as structured data
            defaultOutputModes=["data"],  # Returns trade decision as structured data
        )
    except Exception as e:
        logger.exception("Error creating AgentCard")
        raise

    # Instantiate the AlphaBot Task Manager
    try:
        task_manager = AlphaBotTaskManager()
    except Exception as e:
        logger.exception("Error initializing AlphaBotTaskManager")
        raise

    # Instantiate the A2A Server
    try:
        server = A2AServer(
            agent_card=agent_card,
            task_manager=task_manager,
            host=host,
            port=port,
        )
    except Exception as e:
        logger.exception("Error initializing A2AServer")
        raise

    # Start the Server
    logger.info(f"Starting AlphaBot A2A server on http://{host}:{port}/")
    try:
        server.start()
    except MissingAPIKeyError as e:  # Tool dependency might require keys
        logger.error(f"Configuration Error: {e}")
        exit(1)  # Keep exit for critical config errors
    except Exception as e:
        logger.exception("An error occurred during server startup")
        exit(1)  # Keep exit for critical startup errors


if __name__ == "__main__":
    # Example: python -m agentic_trading.alphabot --port 8081
    main()
