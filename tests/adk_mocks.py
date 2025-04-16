from unittest.mock import MagicMock

# Mocks for ADK classes used when ADK is not installed


class BaseSessionService(MagicMock):
    pass


class Session(MagicMock):
    pass


class InMemorySessionService(BaseSessionService):
    def create_session(self, app_name, user_id):
        return Session()


class InvocationContext:
    def __init__(
        self,
        user_content=None,
        session_service=None,
        invocation_id=None,
        agent=None,
        session=None,
    ):
        # Basic type checks mimicking ADK/Pydantic behavior
        if not isinstance(session_service, BaseSessionService):
            raise TypeError("session_service must be BaseSessionService")
        if not isinstance(session, Session):
            raise TypeError("session must be Session")
        self.user_content = user_content
        self.session_service = session_service
        self.invocation_id = invocation_id
        self.agent = agent
        self.session = session


class genai_types:
    @staticmethod
    def Content(parts):
        return MagicMock(parts=parts)

    @staticmethod
    def Part(text):
        return MagicMock(text=text)


class Event:
    def __init__(self, author, content, actions=None, turn_complete=False):
        self.author = author
        self.content = content
        self.actions = actions
        self.turn_complete = turn_complete

    # Mock get_function_responses: Real ADK Event has this. Agent logic
    # (e.g., AlphaBot processing tool events) might call it. This prevents
    # errors by providing a basic mock structure for function call results.
    def get_function_responses(self):
        # Mimic a function response if author matches alphabot's mock tool.
        if self.author == "MockTool":
            mock_response_part = MagicMock()
            mock_response_part.response = {"approved": True, "reason": "Mock approval"}
            return [mock_response_part]
        return []
