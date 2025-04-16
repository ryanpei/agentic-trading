import logging

import click
import uvicorn

import common.config as defaults
from common.server import A2AServer
from common.types import AgentCapabilities, AgentCard, AgentSkill, MissingAPIKeyError

from .agent import root_agent as riskguard_adk_agent
from .task_manager import RiskGuardTaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the server to.")
@click.option(
    "--port",
    default=defaults.DEFAULT_RISKGUARD_PORT,
    help="Port to bind the server to.",
)
def main(host, port):
    """Runs the RiskGuard ADK agent as a REAL A2A server."""
    logger.info(f"Configuring RiskGuard A2A server...")

    try:
        agent_card = AgentCard(
            name=riskguard_adk_agent.name,
            description=riskguard_adk_agent.description,
            url=f"http://{host}:{port}/",
            version="1.1.0",
            capabilities=AgentCapabilities(
                streaming=False,
                pushNotifications=False,
            ),
            skills=[
                AgentSkill(
                    id="check_trade_risk",
                    name="Check Trade Risk",
                    description="Validates if a proposed trade meets risk criteria.",
                    examples=["Check if buying 100 TECH_STOCK at $150 is allowed."],
                )
            ],
            defaultInputModes=["data"],
            defaultOutputModes=["data"],
        )
    except AttributeError as e:
        logger.error(
            f"Error accessing attributes from riskguard_adk_agent: {e}. Is riskguard/agent.py correct?"
        )
        raise

    try:
        task_manager = RiskGuardTaskManager()
    except Exception as e:
        logger.error(f"Error initializing RiskGuardTaskManager: {e}")
        raise

    try:
        server = A2AServer(
            agent_card=agent_card,
            task_manager=task_manager,
            host=host,
            port=port,
        )
    except Exception as e:
        logger.error(f"Error initializing A2AServer: {e}")
        raise

    # Start the Server
    logger.info(f"Starting RiskGuard A2A server on http://{host}:{port}/")
    try:
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Configuration Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
