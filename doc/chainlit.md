# Chainlit UI

This UI is the interactive front end for the backend agent system.

It sends user turns to the FastAPI backend and renders tool and workflow responses in a chat-oriented format.

Behavior notes:

- incomplete requests may trigger follow-up questions
- new questions can interrupt stale pending state
- chat history is persisted through Chainlit storage
- login identity is reused as stable backend `user_id`
- Chainlit thread id is reused as backend `session_id`

Configuration note:

- backend address is controlled by `BACKEND_URL`

Authentication:

- set `CHAINLIT_AUTH_USERS` or `CHAINLIT_AUTH_USERNAME` / `CHAINLIT_AUTH_PASSWORD`
- set `CHAINLIT_AUTH_SECRET`
