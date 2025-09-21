import json
import os
import uuid
from typing import Type, TypeVar, Optional
from urllib.parse import urlparse
import logging

from pydantic import BaseModel, ValidationError
from google.adk.agents.invocation_context import InvocationContext
from a2a.types import (
    Message,
    Part,
    DataPart,
    Role,
    SendMessageRequest,
    MessageSendParams,
)

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
    if input_text is None:
        logger.error(
            f"[{agent_name} ({invocation_id_short})] ERROR - Input event text is None."
        )
        return None

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


def create_a2a_message_from_payload(
    payload: BaseModel,
    role: Role,
    metadata: Optional[dict] = None,
    context_id: Optional[str] = None,
) -> Message:
    """
    Creates a standardized a2a.types.Message from a Pydantic model payload.
    The payload is serialized and placed into a single DataPart.
    """
    data_part = DataPart(data=payload.model_dump())

    return Message(
        message_id=str(uuid.uuid4()),
        role=role,
        parts=[Part(root=data_part)],
        metadata=metadata or {},
        context_id=context_id,
    )


def create_a2a_request_from_payload(
    payload: BaseModel | DataPart, role: Role = Role.user
) -> SendMessageRequest:
    """
    Creates a complete SendMessageRequest from a Pydantic model payload or a DataPart.
    """
    if isinstance(payload, DataPart):
        data_part = payload
    else:
        data_part = DataPart(data=payload.model_dump())

    message = Message(
        message_id=str(uuid.uuid4()),
        role=role,
        parts=[Part(root=data_part)],
        metadata={},
        context_id=None,
    )
    send_params = MessageSendParams(message=message)
    return SendMessageRequest(id=str(uuid.uuid4()), params=send_params)


def configure_a2a_server_params(
    host_arg: str | None, port_arg: int | None, default_url: str
) -> tuple[str, int, str]:
    """
    Determines the host, port, and base URL for an A2A server.
    Priority: command-line > environment variables > default config.
    """
    # Determine port
    if port_arg:
        port = port_arg
    else:
        default_port = 8080  # Sane fallback
        try:
            parsed_url = urlparse(default_url)
            if parsed_url.port:
                default_port = parsed_url.port
        except Exception:
            logger.warning(
                f"Could not parse default port from URL: '{default_url}'. Using fallback {default_port}."
            )
        port = int(os.environ.get("PORT", default_port))

    # Determine host
    host = host_arg or "0.0.0.0"

    # Determine public-facing base URL for the agent card
    a2a_base_url = os.environ.get("A2A_BASE_URL", f"http://{host}:{port}")

    logger.info(f"A2A Server configured to listen on {host}:{port}")
    logger.info(f"A2A Base URL for Agent Card: {a2a_base_url}")

    return host, port, a2a_base_url
