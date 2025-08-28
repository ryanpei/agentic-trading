import json
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError
import logging

from google.adk.agents.invocation_context import InvocationContext

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


def parse_and_validate_input(
    ctx: InvocationContext, model: Type[T], agent_name: str
) -> T | None:
    """Parses and validates input JSON into a Pydantic model."""
    invocation_id_short = ctx.invocation_id[:8]
    logger.debug(
        f"[{agent_name} ({invocation_id_short})] Attempting to parse input with Pydantic..."
    )
    if (
        not ctx.user_content
        or not ctx.user_content.parts
        or not hasattr(ctx.user_content.parts[0], "text")
    ):
        logger.error(
            f"[{agent_name} ({invocation_id_short})] ERROR - Input event text not found."
        )
        return None

    input_text = ctx.user_content.parts[0].text
    try:
        input_payload = json.loads(input_text)
        validated_input = model(**input_payload)
        logger.info(
            f"[{agent_name} ({invocation_id_short})] Successfully parsed and validated input."
        )
        return validated_input
    except json.JSONDecodeError as e:
        logger.error(
            f"[{agent_name} ({invocation_id_short})] ERROR - Failed to decode input JSON: '{input_text[:200]}...'. Error: {e}"
        )
        return None
    except ValidationError as e:
        logger.error(
            f"[{agent_name} ({invocation_id_short})] ERROR - Input validation failed: {e}. Input was: '{input_text[:200]}...'"
        )
        return None
